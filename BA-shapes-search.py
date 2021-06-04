import numpy as np
import pickle as pkl
from Extractor import Extractor
from scipy.sparse import coo_matrix,csr_matrix
#import tensorflow as tf
import torch
from utils import *
import networkx as nx
from matplotlib import pyplot as plt
from model import *
from train import *

from explain import *
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_networkx
import os
import sys

np.set_printoptions(threshold=sys.maxsize)

dataset_name = 'syn1'
with open('./dataset/' + dataset_name + '.pkl', 'rb') as fin:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix  = pkl.load(fin)
hops = 3
allnodes = [i for i in range(511,871,6)]
all_label = np.logical_or(y_train,np.logical_or(y_val,y_test))
csr_adj = csr_matrix(adj)
extractor = Extractor(csr_adj,features,edge_label_matrix,all_label,hops)
sub_adj,sub_feature, sub_label,sub_edge_label_matrix = extractor.subgraph(allnodes[0])
single_label = np.argmax(all_label,axis=-1)
#len([i for i in range(single_label.shape[0]) if single_label[i] ==1])

class BA_Shape_Dataset(Dataset):
    def __init__(self, root, name, setting = 1, hops=3, transform = None, pre_transform = None, subgraph = False, remap=False):
        super(BA_Shape_Dataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.subgraph = subgraph
        self.remap = remap
        self.name = name
        self.setting = setting
        with open(os.path.join(self.root, name + '.pkl'), 'rb') as fin:
            self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask, self.edge_label_matrix  = pkl.load(fin)
        self.hops = hops

        self.all_label = np.logical_or(self.y_train,np.logical_or(self.y_val,self.y_test))
        self.single_label = np.argmax(self.all_label,axis=-1)
        #self.allnodes = [i for i in range(self.single_label.shape[0]) if self.single_label[i] != 0] #[i for i in range(400,700,5)]
        self.csr_adj = csr_matrix(self.adj)
        self.extractor = Extractor(self.csr_adj,self.features,self.edge_label_matrix,self.all_label,self.hops)            
    @property
    def num_features(self):
        return 10
    @property
    def num_classes(self):
        return self.all_label.shape[1]

    @property 
    def allnodes(self):
        if self.setting==1:
            if self.name=='syn3':
                allnodes = [i for i in range(511,871,6)]
            elif self.name=='syn4':
                allnodes = [i for i in range(511,800,1)]
            else:
                allnodes = [i for i in range(400,700,5)] # setting from their original paper
        elif self.setting==2:
            allnodes = [i for i in range(self.single_label.shape[0]) if self.single_label[i] ==1]
        elif self.setting==3:
            if self.name == 'syn2':
                allnodes = [i for i in range(self.single_label.shape[0]) if self.single_label[i] != 0 and self.single_label[i] != 4]
            else:
                allnodes = [i for i in range(self.single_label.shape[0]) if self.single_label[i] != 0]
        return allnodes


    def get_subgraph(self, idx):
        #if self.subgraph:
        if True:
            idx = self.allnodes[idx] if self.remap else idx
            sub_adj,sub_feature, sub_label,sub_edge_label_matrix = dataset.extractor.subgraph(idx)
            return sub_adj,sub_feature, sub_label,sub_edge_label_matrix
        #else:
        #    return None
    def len(self):
        if self.subgraph:
            if not self.remap:
                return len(self.single_label)
            return len(self.allnodes)
        else:
            return 1
    
    def get(self, idx):
        if self.subgraph:
            if self.remap:
                idx = self.allnodes[idx]
            sub_adj,sub_feature, sub_label,sub_edge_label_matrix = self.extractor.subgraph(idx)
            edge_index = torch.tensor(preprocess_adj(sub_adj)[0].T, dtype = torch.long)
            x = torch.tensor(sub_feature).float()
            y = torch.argmax(torch.tensor(sub_label, dtype = torch.int32), dim=-1)
            data = Data(edge_index = edge_index, x = x, y = y)
        else:

            edge_index = torch.tensor(preprocess_adj(self.adj)[0].T, dtype = torch.long)
            x = torch.tensor(self.features).float()
            y = torch.argmax(torch.tensor(np.logical_or(self.y_train,np.logical_or(self.y_val,self.y_test)), dtype = torch.int32), dim=-1)
            data = Data(edge_index = edge_index, x = x, y = y)
        return data

dataset = BA_Shape_Dataset(root = './dataset', name = 'syn1', setting = 3)

for i in range(1):
    gcn_model = ATTGCN_Node(dataset, 3, 20)
    #gcn_model = GCN_Node(dataset, 3, 20)
    dataset.setting = 3
    dataset.subgraph = False
    dataset.remap = True
    node_pred_task_train(gcn_model, dataset, max_epoch=150, lr=0.005, temp_name='shape_temp', train_rate = 0.8)

    gcn_model = torch.load('./checkpoint/shape_temp')

    acc_list = []
    auc_list = []

    dataset.subgraph = False
    dataset.remap = False
    dataset.setting=3
    load_model = gcn_model
    #load_model = torch.load('./checkpoint/gcn_mix').to('cuda')
    load_model.eval()

    all_node_label = []
    all_node_color = []
    all_elapsed = []
    all_len = []
    for idx in dataset.allnodes:
        #idx = 313
        print('\n=================index: ', idx)
        sub_adj,sub_feature, sub_label,sub_edge_label_matrix = dataset.get_subgraph(idx)
        #truth_node = np.where(sub_label[:,1] == True)[0]
        truth_node = list(get_node_set(sub_edge_label_matrix))
        
        class_idx = np.argmax(sub_label[0],axis=-1)
        print('0 label: ', class_idx)
        node_range = dataset.extractor.nodes
        #print(node_range)
        node_sort, node_color, elapsed = print_explain(dataset = dataset, model = load_model, idx = 0, class_idx = class_idx, visible = False, figsize = (12,9), node_range = node_range)
        #print(node_sort)
        all_elapsed.append(elapsed)
        all_len.append(len(node_color))

        node_label = np.array([0] * sub_label.shape[0])
        #node_label[list(truth_node)] = 1
        # find truth node, far node is not real truth
        for n in truth_node:
            if abs((node_range[n] - node_range[0])) <= 8:
                node_label[n] = 1
                #print(n)

        try:
            auc = roc_auc_score(node_label, node_color)
        except:
            print('foo')
            continue
            auc = 1.0

        #print("truth node: ", truth_node)
        #print(node_sort)
        acc = len([node for node in node_sort[:5] if node in truth_node])/5
        acc_list.append(acc)
        auc_list.append(auc)
        #all_node_label.extend(node_label)
        #all_node_color.extend(node_color)
        print('acc: ', acc)
        print('auc: ', auc)
        #if acc == 0.0:
        #    print(node_sort)
        #    print_explain(dataset, load_model, idx, class_idx = np.argmax(sub_label[0],axis=-1), visible = True)
        print('mean acc: ', np.mean(acc_list))
        print('mean auc: ', np.mean(auc_list))
        #break
    import pickle
    with open('time_shape_gat', 'wb') as f:
        pickle.dump({'all_elapsed': all_elapsed, 'all_len':all_len}, f)
    torch.save(load_model, './BA_shape_search_checkpoint_GAT_3/acc_' + str(np.mean(acc_list)) + '_auc_' + str(np.mean(auc_list)))