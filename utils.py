import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from torch_geometric.data import Dataset, Data, DataLoader

def get_data(dataset, idx, batch_size=1, shuffle = False):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    for batch_idx, data in enumerate(loader):
        if not batch_idx == idx:
            continue
        return data

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        values = values.astype(np.float32)
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_adj(adj, norm=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if norm:
        adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
        return sparse_to_tuple(adj_normalized)
    else:
        return sparse_to_tuple(sp.coo_matrix(adj))

def get_node_set(sub_edge_label_matrix):
    node_set = set()
    for e in preprocess_adj(sub_edge_label_matrix)[0][np.where(preprocess_adj(sub_edge_label_matrix)[1] == 1)]:
        node_set.add(e[0])
        node_set.add(e[1])
    return node_set