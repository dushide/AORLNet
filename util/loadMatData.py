import os
from util.hypergraph_utils import construct_H_with_KNN, generate_G_from_H
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse as ss
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
def load_data(args, dataset):
    data = sio.loadmat(args.data_path + dataset + '.mat')
    feature = data['X']
    labels = data['Y'].flatten()
    labels = labels - min(set(labels))
    feature = normalize(feature)
    adj = data['adj']
    return feature, labels, adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def construct_hypergraph(dataset,feature, knn,device):
    direction_judge = './hyperlap_matrix/' + dataset + '/' + 'knn' + str(knn) + '_hyperlap.npz'
    direction_judge2 = './hyperlap_matrix/' + dataset + '/'+ 'knn' + str(knn) + '_hyperadj.npz'
    if os.path.exists(direction_judge) and os.path.exists(direction_judge2):
        print("Loading the hyperlap matrix of " + dataset)
        temp_lap = ss.load_npz(direction_judge)
        lap=torch.from_numpy(temp_lap.todense()).float().to(device)

        temp_adj = ss.load_npz(direction_judge2)
        adj=torch.from_numpy(temp_adj.todense()).float().to(device)
    else:
        print("Constructing the hyperlap matrix of "  + dataset)
        H = construct_H_with_KNN([feature], knn, split_diff_scale=True)
        G,temp_adj = generate_G_from_H(H)
        temp_lap = np.identity(len(G)) - G
        save_direction = './hyperlap_matrix/' + dataset + '/'
        if not os.path.exists(save_direction):
            os.makedirs(save_direction)
        print("Saving the adjacency matrix to " + save_direction)
        ss.save_npz(direction_judge , ss.csr_matrix(temp_lap[0]))
        ss.save_npz(direction_judge2, ss.csr_matrix(temp_adj[0]))
        lap=torch.from_numpy(temp_lap[0]).to(torch.float32).to(device)
        adj=torch.from_numpy(temp_adj[0]).to(torch.float32).to(device)
    return lap,adj
def construct_hypergraph_noise(dataset,feature, knn,device):

    print("Constructing the hyperlap matrix of "  + dataset)
    H = construct_H_with_KNN([feature], knn, split_diff_scale=True)
    G,temp_adj = generate_G_from_H(H)
    temp_lap = np.identity(len(G)) - G
    lap=torch.from_numpy(temp_lap[0]).to(torch.float32).to(device)
    adj=torch.from_numpy(temp_adj[0]).to(torch.float32).to(device)
    return lap,adj
def construct_laplacian(adj):
    """
        construct the Laplacian matrix
    :param adj: original Laplacian matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj_ = sp.coo_matrix(adj)
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())  # degree matrix
    adj_wave = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    lp = sp.eye(adj.shape[0]) - adj_wave
    return lp

def features_to_Lap(features, knns=10):
    temp = kneighbors_graph(features, knns)
    temp = sp.coo_matrix(temp)
    # build symmetric adjacency matrix
    temp = temp + temp.T.multiply(temp.T > temp) - temp.multiply(temp.T > temp)
    lap=sparse_mx_to_torch_sparse_tensor(construct_laplacian(temp))
    return lap