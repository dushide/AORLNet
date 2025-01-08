from __future__ import print_function, division

from util.loadMatData import construct_hypergraph
import random
import argparse
import sys
from cluster import cluster
from util.config import load_config
import numpy as np
import torch
from util.loadMatData import load_data


def main(data, args):
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    config = load_config(r'./config/Hyp-GSpNet.yaml')

    feature, labels, _ = load_data(args, data)
    n = feature.shape[0]
    n_feats = feature.shape[1]
    n_clusters = len(np.unique(labels))



    args.k=int(n / n_clusters)
    lap_Z, adj = construct_hypergraph(data, feature, args.k, device)

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
    parser.add_argument("--save_path", type=str, default="./result/Hyp-GSpNet.txt",
                        help="Save experimental result.")

    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--fix_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 40.")

    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--thre1', type=float, default=0.1)
    parser.add_argument('--thre2', type=float, default=0.01)
    parser.add_argument('--delta', type=float, default=1, help='delta')
    parser.add_argument('--block', type=int, default=1, help='The block number')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--active_O', type=str, default="l21")

    args = parser.parse_args()
    SCORE = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']
    dataset = {1: "Chameleon", 2: "film", 3: "Squirrel", 4: "Tesax", 5: "Wiki", 6: "Wisconsin"}

    select_dataset = [1, 2, 3, 4, 5, 6]

    for i in select_dataset:
        args.data = dataset[i]
        main(args.data, args)
