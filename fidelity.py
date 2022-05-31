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


np.set_printoptions(threshold=sys.maxsize)
from torch_geometric.datasets import TUDataset, Entities
from torch_geometric.data import DataLoader
dataset = TUDataset(root='./dataset/Mutagenicity', name='Mutagenicity', use_node_attr=True)

train_len = int(0.8 * dataset.len())
test_len = dataset.len() - train_len
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
gcn_model = torch.load('./MUTAG_search_checkpoint_2/auc_0.9326872275822641')
load_model = gcn_model

kwargs = {}
kwargs['node_name_list'] = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'K', 'Li', 'Ca']
select = np.load('gt_index.npy')
edge_lists = np.load('edge_lists.npy', allow_pickle = True)
edge_label_lists = np.load('edge_label_lists.npy', allow_pickle=True)

#new version without mean middle
def adjacent_process(data, node_range = None):
    '''
    transfer an edge_index to matrix hash map
    node_range: node list that is considered
    '''
    node_range = range(data.x.shape[0]) if node_range is None else node_range
    matrix = {}
    for n in node_range:
        matrix[n] = []
    

    for i in range(data.edge_index.shape[1]):
        r = int(data.edge_index[0][i])
        c = int(data.edge_index[1][i])
        if r not in node_range or c not in node_range:
            continue
        matrix[ r ].append( c )
    #print('=========A: ', matrix)
    return matrix

def if_connect(set_1, set_2, A):
    '''
    return if two sets connect to each other
    A is adjacent matrix hash map
    '''
    for i in set_1:
        for j in set_2:
            if j in A[i]:
                return True

    return False

def find_neighbor(node_list, A):
    '''
    find neighbor of node_list
    A : adjacent matrix
    '''

    ans = []
    for n in node_list:
        for ne in A[n]:
            if ne not in node_list and ne not in ans:
                ans.append(ne)

    return ans

def merge(comp_list, cand_list, A):
    '''
    merge new candidates node to old comp, aggregate linked group
    comp_list like: [[1,2], [3,4,5], [6]]
    cand_list like: [7,8,9]
    A: adjacent matrix hash map
    '''
    for cand in cand_list:
        new_comp_list = [[cand]]
        for comp in comp_list:
            if if_connect(new_comp_list[0], comp, A):
                new_comp_list[0].extend(comp)
            else:
                new_comp_list.append(comp)

        comp_list = new_comp_list

    return comp_list

def get_rw_score(load_model, data, class_idx, comp, A,softmax):
    # get neighbor
    #print('======= comp is:', comp)
    n_list = find_neighbor(comp, A)
    #print('======= its neighbor is ', n_list)
    score = get_score(load_model, data.to(device), [comp], softmax = softmax)[class_idx]['rel'][0]
    if len(n_list) == 0:
        return score
    else:
        for n in n_list:
            score += get_score(load_model, data.to(device), [comp + [n]], softmax = softmax)[class_idx]['rel'][0]
        return score / (len(n_list)+1)
def agglomerate(load_model, data, percentile = 60, class_idx = 0, node_range = None, softmax = False):
    class_idx = class_idx if class_idx is not None else (int)(dataset[idx].y[0])
    #print('======= class idx: ', class_idx)
    node_range = range(data.x.shape[0]) if node_range is None else node_range
    
    A = adjacent_process(data, node_range)
    # init build, every node score
    mask_list = []
    for i in node_range:
        mask_list.append([i])
    scores = []
    for m in mask_list:
        scores.append(get_rw_score(load_model, data, class_idx, m, A, softmax=softmax))
    #scores = get_score(load_model, data.to('cuda'), mask_list, softmax = softmax)[class_idx]['rel']
    #print('++++',scores)
    # mean sub
    #print('before scores: ', scores)
    #scores = scores - np.mean(scores)
    #scores = np.absolute(scores)
    #print('after scores: ', scores)

    thresh_hold = np.nanpercentile(scores, percentile)
    #cand_list = np.where(scores >= thresh_hold)[0].tolist()
    cand_list = np.take(node_range, np.where(scores >= thresh_hold)[0])
    cand_list = list(set(cand_list) & set(node_range))
    
    comp_list = merge([], cand_list, A)
    #print('comp_list: ', comp_list)

    # begin add nodes
    comp_all = []
    comp_all_score = []

    comp_all.append(mask_list.copy())
    #comp_all_score.append([ get_score(load_model, data.to('cuda'), [c], softmax = False)[class_idx]['rel'][0] for c in mask_list])
    comp_all_score.append([ get_rw_score(load_model, data, class_idx, c, A, softmax=softmax) for c in mask_list])

    comp_all.append(comp_list.copy())
    comp_all_score.append([ get_rw_score(load_model, data, class_idx, c, A, softmax=softmax) for c in comp_list])

    for _ in tqdm(range(500)):
        print('###############loop: ', _ )
        # find neibors for each comp
        cand_list = []
        scores = []
        for comp in comp_list:
            neighbors = find_neighbor(comp, A)
            #print('comp ',comp, " : ", find_neighbor(comp, A))
            # score for each neighbor
            for n in neighbors:
                old_set = comp.copy()
                new_set = comp.copy()
                new_set.append(n)
                #print('old_Set: ', old_set)
                #print('new_Set: ', new_set)

                #scores.append(get_score(load_model, data.to('cuda'), [new_set], softmax = False)[class_idx]['rel'][0] -  get_score(load_model, data.to('cuda'), [old_set], softmax = False)[class_idx]['rel'][0])
                scores.append(get_rw_score(load_model, data, class_idx, new_set, A, softmax=softmax) -  get_rw_score(load_model, data, class_idx, old_set, A, softmax=softmax))

            cand_list.extend(neighbors)
        #print('cand_list: ', cand_list)
        #print('scores: ', scores)
        #scores -= np.mean(scores)

        #scores = np.absolute(scores)
        thresh_hold = np.nanpercentile(scores, percentile)
        #print('thresh_hold: ', thresh_hold)
        cand_list = np.take(cand_list, np.where(scores >= thresh_hold)[0])
        cand_list = np.unique(cand_list)
        #print(cand_list)
        #print(comp_list)
        # merge 
        comp_list = merge(comp_list, cand_list, A)
        #print(comp_list)
        comp_sum = np.sum([ len(c) for c in comp_list])
        #print([ len(a) for c in comp_list])
        #print('comp sum: ', comp_sum)
        comp_all.append(comp_list.copy())
        #comp_all_score.append([ get_score(load_model, data.to('cuda'), [c], softmax = False)[class_idx]['rel'][0] for c in comp_list])
        comp_all_score.append([ get_rw_score(load_model, data, class_idx, c, A, softmax=softmax) for c in comp_list])
        if len(node_range) <= comp_sum or len(cand_list)==0:
            #print('BBBBBBBBBBBBBRRRRRRRRRRRRRRREEEEEEEEEEEEEEKKKKKKKKKKKKKK')
            break

    #print('comp_all_score: ', comp_all_score)
    return comp_all, comp_all_score

def hier_explain(dataset, model, idx, class_idx = None, visible = False, figsize = None, percentile = 75, **kwargs):
    global edges
    class_idx = int(class_idx) if class_idx is not None else (int)(dataset[idx].y[0])
    idx = int(idx)
    
    data = get_data(dataset, idx)
    begin_time = time.time()
    comp_all,comp_all_score = agglomerate(model, data, percentile = percentile)
    duration = time.time()-begin_time
    color_max = max([max(comp_score) for comp_score in comp_all_score]) 
    color_min = min([min(comp_score) for comp_score in comp_all_score]) 
    #color_max = 0.8
    #color_min = -0.8
    print('color range: ', color_max, ' ', color_min)
    node_sizes = 1000
    if visible:

        G = to_networkx(dataset[idx])

        pos = nx.kamada_kawai_layout(G)

        #node_sizes = 600 * size_ratio
        #node_colors = class_score[class_idx]['rel']


        cmap = plt.cm.coolwarm
        if figsize is None:
            fig = plt.figure(figsize = (10 * len(comp_all),10),dpi=100, facecolor="white")
        else:
            fig = plt.figure(figsize = figsize ,dpi=100, facecolor="white")
        
        for layer in range(len(comp_all)):
            plt.subplot(1, len(comp_all), layer+1)

            #find white nodes
            white_nodes = set([n for n in range(data.x.shape[0])])
            comp_list = comp_all[layer]
            comp_score = comp_all_score[layer]


            # print comp nodes
            for i in range(len(comp_list)):
                white_nodes = white_nodes - set(comp_list[i])

                nodes = nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist = comp_list[i],
                    node_size = node_sizes,
                    node_color = [comp_score[i]] * len(comp_list[i]),
                    cmap=cmap,
                    #edgecolors='blue',
                    vmax = color_max,
                    vmin = color_min,
                    alpha = 0.6,
                )


                edges = nx.draw_networkx_edges(
                    G.subgraph(comp_list[i]),
                    pos,
                    #node_size=node_sizes,
                    arrowstyle="-",
                    #arrowsize=10,
                    width=20,
                    connectionstyle= 'arc3, rad=0.2',
                    style='-.',
                    alpha=0.3,
                    edge_cmap = cmap,
                    edge_color = [comp_score[i]] * len(G.subgraph(comp_list[i]).edges),
                    edge_vmin = color_min,
                    edge_vmax = color_max,

                    
                )
            
            # print white nodes
            nodes_ = nx.draw_networkx_nodes(
                G,
                pos,
                nodelist = list(white_nodes),
                node_size = node_sizes,
                node_color = ['tab:gray'] * len(white_nodes),
                cmap=cmap,
                #edgecolors='blue',
                vmax = color_max,
                vmin = color_min,
                alpha = 0.3
            )

            edges = nx.draw_networkx_edges(
                G,
                pos,
                #node_size=node_sizes,
                arrowstyle="-",
                #arrowsize=10,
                width=2,
                #connectionstyle= 'arc, rad=0.2',
                style=':',
                alpha=0.3
            )
            label_list = {}
            for i in range(dataset[idx].x.shape[0]):
                label_list[i] = str(i)
                if 'node_name_list' in kwargs:
                    node_name_idx = int(np.argmax(dataset[idx].x[i]))
                    label_list[i] += '' + kwargs['node_name_list'][node_name_idx]

            labels = nx.draw_networkx_labels(G, pos, label_list, font_size=12, font_color="black")
            #plt.colorbar(nodes)

            #ax = plt.gca()
            #ax.set_axis_off()
            #if layer+1 == 1:
            #    continue
            #plt.subplot(1, len(comp_all), layer+1)
            print("layer", layer+1)
        plt.colorbar(nodes)
        plt.show()
    return duration, data.x.shape[0], comp_all

idx_list = np.random.choice(select, 1000)
comp_all_list = []

sparsity_all_list = []
fidelity_all_list = []
reverse_sparsity_all_list = []
reverse_fidelity_all_list = []
statistic = [[],[],[],[],[],[],[],[],[],[]]
reverse_statistic = [[],[],[],[],[],[],[],[],[],[]]

for i in tqdm(idx_list):
    data = get_data(dataset, i)
    #print(data.x.shape[0])
    if(data.x.shape[0]> 200): continue
    _,_,comp_all = hier_explain(dataset, load_model, idx = i, class_idx = 0,visible = False, figsize = None, percentile = 80, **kwargs)
    comp_all_list.append(comp_all)


#for (idx, comp_all) in tqdm(zip(idx_list, comp_all_list)):
    idx = i
    mask_list = []
    sparsity_all = []
    fidelity_all = []
    
    reverse_sparsity_all = []
    reverse_fidelity_all = []

    data = get_data(dataset, idx).to('cuda')
    pred = load_model(data)
    
    for c in tqdm(comp_all[1:]):
        mask = [0]*data.x.shape[0]
        for m in c:
            for mm in m:
                mask[mm] = 1
        #print(np.sum(mask)/data.x.shape[0])
        sparsity_all.append(np.sum(mask)/data.x.shape[0])
        mask = torch.tensor([mask])

        m = mask.T.to('cuda')


        data = get_data(dataset, idx).to('cuda')
        data.x *= m
        if (load_model(data).argmax() == pred.argmax()):
            fidelity_all.append(1.0)
        else:
            fidelity_all.append(0.0)

    for c in tqdm(comp_all[1:]):
        mask = [1]*data.x.shape[0]
        for m in c:
            for mm in m:
                mask[mm] = 0
        #print(np.sum(mask)/data.x.shape[0])
        reverse_sparsity_all.append(1 - np.sum(mask)/data.x.shape[0])
        mask = torch.tensor([mask])

        m = mask.T.to('cuda')


        data = get_data(dataset, idx).to('cuda')
        data.x *= m
        if (load_model(data).argmax() == pred.argmax()):
            reverse_fidelity_all.append(1.0)
        else:
            reverse_fidelity_all.append(0.0)

            
    sparsity_all_list.append(sparsity_all)
    fidelity_all_list.append(fidelity_all)
            
    reverse_sparsity_all_list.append(reverse_sparsity_all)
    reverse_fidelity_all_list.append(reverse_fidelity_all)

    print('sparsity_all_list: ', sparsity_all_list)
    print('fidelity_all_list: ', fidelity_all_list)
    print('reverse_sparsity_all_list: ', reverse_sparsity_all_list)
    print('reverse_fidelity_all_list: ', reverse_fidelity_all_list)

    statistic = [[],[],[],[],[],[],[],[],[],[]]
    reverse_statistic = [[],[],[],[],[],[],[],[],[],[]]

    for (sparsity_all, fidelity_all) in zip(sparsity_all_list,fidelity_all_list):
        #print(fidelity_all)
        for (sparsity, fidelity) in zip(sparsity_all, fidelity_all):
            #print(fidelity)
            idx = int(sparsity * 10) - 1
            
            statistic[idx].append(fidelity)

    for (reverse_sparsity_all, reverse_fidelity_all) in zip(reverse_sparsity_all_list,reverse_fidelity_all_list):
        #print(fidelity_all)
        for (reverse_sparsity, reverse_fidelity) in zip(reverse_sparsity_all, reverse_fidelity_all):
            #print(fidelity)
            idx = int(reverse_sparsity * 10) - 1
            
            reverse_statistic[idx].append(reverse_fidelity)

    import pickle
    with open("fidelity_statistic", 'wb') as f:
        pickle.dump(statistic,f)

    with open('fidelity_statistic', 'rb') as f:
        statistic = pickle.load(f)

    with open("fidelity_statistic_reverse", 'wb') as f:
        pickle.dump(reverse_statistic,f)

    with open('fidelity_statistic_reverse', 'rb') as f:
        reverse_statistic = pickle.load(f)

    for s in statistic:
        print(np.mean(s))

    for s in reverse_statistic:
        print(np.mean(s))

        