from tqdm import tqdm
from util.clusteringPerformance import StatisticClustering
from model import CombineNet
import torch
import torch.nn as nn
import time
import numpy as np

SCORE = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']


def cluster(args, device, data, n, n_feats, n_clusters, feature, adj, lap, labels):
    model = CombineNet(n, n_feats, n_clusters, args, device,
                       ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.90, 0.92), eps=0.01, weight_decay=0.15)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=15, verbose=True,
                                                           min_lr=1e-8)
    res_list = list()
    criterion = nn.MSELoss()
    start = time.perf_counter()
    with tqdm(total=args.epoch, desc="Training") as pbar:
        for i in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            output_z = model(feature, lap, args.active_O)
            loss_sem = criterion(output_z.mm(output_z.t()), feature.mm(feature.t()))
            loss_top = criterion(output_z.mm(output_z.t()), adj)

            loss = loss_sem + args.gamma * loss_top
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            Sem_loss = loss_sem.cpu().detach().numpy()
            Top_loss = loss_top.cpu().detach().numpy()
            train_loss = loss.cpu().detach().numpy()
            print({"Self1 loss": "{:.6f}".format(Sem_loss),
                   "Self2 loss": "{:.6f}".format(Top_loss),
                   'Loss': '{:.6f}'.format(train_loss)})

            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(output_z.detach().cpu().numpy(),
                                                                                     labels, n_clusters)
            res_list.append([ACC, NMI, Purity, ARI, Fscore, Precision, Recall])
            pbar.update(1)
    end = time.perf_counter()
    total_time = (end - start)
    print("time : {:.2f}".format(total_time))

    res_list.sort(key=lambda item: item[0][0], reverse=True)
    res = list()
    for item in res_list[0]:
        res.append(f"{item[0] * 100:.2f} ({item[1] * 100:.2f})")
    res.append(np.around(total_time, 2))

    final_Res = "{}:{}\n".format(args.data, dict(
        zip(SCORE + ['time'], res)))
    with open(args.save_path, "a") as f:
        f.write(final_Res)
