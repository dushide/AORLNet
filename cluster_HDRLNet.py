from __future__ import print_function, division

import random
import time
import argparse
import sys
from util.config import load_config
from tqdm import tqdm
from util.clusteringPerformance import StatisticClustering
import numpy as np
import torch
import torch.nn as nn
from util.loadMatData import load_data, construct_hypergraph
from model import CombineNet


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

            [ACC, NMI, Purity, ARI, Fscore, Precision, Recall] = StatisticClustering(output_z.detach().cpu().numpy(), labels, n_clusters)
            res_list.append([ACC, NMI, Purity, ARI, Fscore, Precision, Recall])
            pbar.update(1)
    end = time.perf_counter()
    total_time = (end - start)
    print("time : {:.2f}".format(total_time))

    res_list.sort(key=lambda item: item[0][0], reverse=True)
    res = list()
    for item in res_list[0]:
        res.append(str(item[0] * 100) + "(" + str(item[1] * 100) + ")")
    res.append(np.around(total_time,2))
    with open(args.save_path, "a") as f:
        f.write("{}:{}\n".format(args.data, dict(
            zip(SCORE + ['time'], res))))


def main(data, args):
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    config = load_config(r'./config/HDRLNet.yaml')

    feature, labels = load_data(args, data)
    n = feature.shape[0]
    n_feats = feature.shape[1]
    n_clusters = len(np.unique(labels))
    lap_Z, adj = construct_hypergraph(data, feature, int(n / n_clusters), device)

    feature = torch.from_numpy(feature / 1.0).float().to(device)

    args.block = config[data]
    if args.fix_seed:
        torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    cluster(args, device, data, n, n_feats, n_clusters, feature, adj, lap_Z, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Parameter setting
    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)
    parser.add_argument("--data_path", type=str, default="./data/", help="Path of datasets.")
    parser.add_argument("--save_path", type=str, default=current_dir + "/HDRLNet_res.txt",
                        help="Save experimental result.")

    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--fix_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 40.")

    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--thre1', type=float, default=0.1)
    parser.add_argument('--thre2', type=float, default=0.1)
    parser.add_argument('--delta', type=float, default=1, help='delta')
    parser.add_argument('--block', type=int, default=1, help='The block number')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--active_O', type=str, default="l1")

    args = parser.parse_args()
    SCORE = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']
    dataset = {1: "Chameleon", 2: "film", 3: "Squirrel", 4: "Tesax", 5: "Wiki", 6: "Wisconsin"}

    select_dataset = [1, 2, 3, 4, 5, 6]

    for i in select_dataset:
        args.data=dataset[i]
        main(args.data, args)
