import sys
import time

import pickle as pkl
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np


from Extractor import Extractor
from scipy.sparse import coo_matrix,csr_matrix
#import tensorflow as tf
import torch
from utils import *
from model import *
from train import *

from explain import *
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_networkx

class ATTGCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, num_features = None):
        super(ATTGCN, self).__init__()
        num_features = dataset.num_features if num_features is None else num_features
        
        self.conv1 = CD_GATConv(in_channels = num_features, out_channels = (int)(hidden/1), heads=1)#CD_GCNConv(num_features, hidden)
       
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(CD_GATConv(in_channels = hidden, out_channels = (int)(hidden/1), heads=1))
        
        self.lin1 = CD_Linear(hidden, hidden)
        
        self.lin2 = CD_Linear(hidden, dataset.num_classes)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, CD_explain: bool = False, mask_index = None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if CD_explain:
            x = mask_x(x, mask_index)
        x = CD_relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = CD_relu(conv(x, edge_index))
        x = CD_global_max_pool(x, batch)
        #x = CD_feature_max_pool(x, batch)
        #print(x.shape)
        x = CD_relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        if CD_explain:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

# dataset  prepare

from torch_geometric.datasets import TUDataset, Entities
from torch_geometric.data import DataLoader
dataset = TUDataset(root='./dataset/Mutagenicity', name='Mutagenicity', use_node_attr=True)
train_len = int(0.8 * dataset.len())
test_len = dataset.len() - train_len
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])


select = np.load('gt_index.npy')
edge_label_lists = np.load('edge_label_lists.npy', allow_pickle=True)

kwargs = {}
kwargs['node_name_list'] = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
edge_lists = np.load('edge_lists.npy', allow_pickle = True)

for _ in range(1):
    # train model
    gcn_model = ATTGCN(dataset, 3, 20)
    #gcn_model = GCN(dataset,3,20)
    train(gcn_model, train_dataset, test_dataset, max_epoch = 10, lr=0.005, save_model = True, temp_name = 'gcn_mutag_temp')
    gcn_model = torch.load('./checkpoint/gcn_mutag_temp')


    acc_list = []
    auc_list = []
    node_num_list = []

    load_model = gcn_model
    #load_model = torch.load('./checkpoint/gcn_mix').to('cuda')
    load_model.eval()

    all_node_label = []
    all_node_color = []
    all_elapsed = []
    all_len = []
    #for idx in range(dataset.len()):
    #for idx in range():
    for idx in range(1015):
        #idx  = 21
        #if idx in in_correct:
        #    continue
        idx = int(select[idx])
        print('\nindex: ', idx)
        data = dataset[idx]
        #truth_node = np.where(sub_label[:,1] == True)[0]
        #truth_node = get_node_set(sub_edge_label_matrix)
        #if len(truth_node) > 6:
        #    continue
        #print('label: ', data.y[0])
        #node_sort, node_color = print_explain(dataset, load_model, idx, class_idx = 0, visible = False, figsize = (8,6), **kwargs)
        #_, _, adj, edge_pred = find_edge(load_model, dataset, idx, class_idx = None, node_sort = node_sort, topk = 10, start_num=1)
        #print(_)
        edge_score, elapsed = Edge_explain(model = load_model, dataset = dataset, edge_list = edge_lists[idx], idx = idx)
        edge_colors = []
        for i in range(len(edge_score[0]['rel'])):
            edge_colors.append( edge_score[0]['rel'][i] - edge_score[1]['rel'][i] )
        #print('edge sort: ', np.argsort(edge_colors)[::-1])
        auc = roc_auc_score(edge_label_lists[idx],edge_colors)
        all_elapsed.append(elapsed)
        all_len.append(len(edge_colors))
        #print('auc: ', auc)
        auc_list.append(auc)
        print('auc mean: ', np.mean(auc_list))
        #print(roc_auc_score(edge_label_lists[idx][0:-1:2], edge_pred))
        #edge_type(adj, data)
        #node_label = np.array([0] * sub_label.shape[0]

    print('loop ', _ ,' finish, saving model...')
    mean_auc = np.mean(auc_list)
    import pickle
    with open('time_mutag_gat', 'wb') as f:
        pickle.dump({'all_elapsed': all_elapsed, 'all_len':all_len}, f)
    torch.save(load_model, './MUTAG_search_checkpoint_GAT_3/auc_' + str(mean_auc))