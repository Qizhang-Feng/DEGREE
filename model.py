import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, JumpingKnowledge, BatchNorm
from CD_layers import CD_GCNConv, CD_global_max_pool, CD_global_mean_pool, CD_Linear, CD_relu, CD_leaky_relu, CD_softmax, CD_feature_max_pool, CD_feature_mean_pool
def mask_x(x, mask_index):
    """
    produce mask
    x input data
    mask_index like [0, 1, 0, 0, 1, ....]
    return rel and irrel of x 
    """
    mask_index = torch.tensor([mask_index]).T.to(x.device)
    result = {}
    result['rel'] = x * mask_index
    result['irrel'] = x * (1-mask_index)
    #print(result)
    return result


class GCN_MUTAG(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GCN_MUTAG, self).__init__()
        #self.dataset = dataset
        self.conv1 = CD_GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(CD_GCNConv(hidden, hidden))

        #self.lin1 = CD_Linear(hidden, hidden)
        self.lin1 = CD_Linear(418, 418)
        self.lin2 = CD_Linear(418, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, CD_explain: bool = False, mask_index = None, Intermediate = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if CD_explain:
            x = mask_x(x, mask_index)
        
        x = self.conv1(x, edge_index)
        x = CD_relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = CD_relu(x)

        x = CD_feature_max_pool(x, batch, batch_size = 418)
        x = self.lin1(x)
        x = CD_relu(x)
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

class GCN_2motif(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GCN_2motif, self).__init__()
        #self.dataset = dataset
        self.conv1 = CD_GCNConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(CD_GCNConv(hidden, hidden))

        #self.lin1 = CD_Linear(hidden, hidden)
        self.lin1 = CD_Linear(25, 25)
        self.lin2 = CD_Linear(25, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, CD_explain: bool = False, mask_index = None, Intermediate = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if CD_explain:
            x = mask_x(x, mask_index)
        
        x = self.conv1(x, edge_index)
        x = CD_relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = CD_relu(x)

        x = CD_feature_max_pool(x, batch)
        x = self.lin1(x)
        x = CD_relu(x)
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


class GCN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, num_features = None, dense = False, concate = False):
        super(GCN, self).__init__()
        num_features = dataset.num_features if num_features is None else num_features
        #self.dataset = dataset
        
        self.dense = dense
        self.concate = concate

        self.convs = torch.nn.ModuleList()
        self.convs.append(CD_GCNConv(dataset.num_features, hidden))
        for i in range(num_layers - 1):
            self.convs.append(CD_GCNConv(hidden, hidden))

        if dense:
            self.bns = torch.nn.ModuleList()
            for i in range(num_layers-1):
                #self.bns.append(CD_Linear(hidden, hidden))
                self.bns.append(torch.nn.BatchNorm1d(hidden))
        if self.concate:
            self.lin1 = CD_Linear(hidden * num_layers, hidden * num_layers)
            self.lin2 = CD_Linear(hidden * num_layers, dataset.num_classes)
        else:
            self.lin1 = CD_Linear(hidden, hidden)
            self.lin2 = CD_Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, CD_explain: bool = False, mask_index = None, Intermediate = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        #print(batch)

        if CD_explain:
            self.eval()
            x = mask_x(x, mask_index)
        
        self.all_x = [] if not CD_explain else {'rel':[], 'irrel':[]}
        for idx in range(len(self.convs)):
            x = self.convs[idx](x, edge_index)
            x = CD_relu(x)

            if self.dense and idx != len(self.convs) -1:
                x = self.bns[idx](x)

            if self.concate:
                if not CD_explain:
                    self.all_x.append(x)
                else:
                    self.all_x['rel'].append(x['rel'])
                    self.all_x['irrel'].append(x['irrel'])
        #print(self.all_x)
        if self.concate:
            if CD_explain:
                x['rel'] = torch.cat(self.all_x['rel'], -1)
                x['irrel'] = torch.cat(self.all_x['irrel'], -1)

            else:    
                x = torch.cat(self.all_x, -1)
        x = CD_global_max_pool(x, batch)
        x = self.lin1(x)
        x = CD_relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        #print(x[:1].shape)
        #print(self.dataset.subgraph)
        #if self.dataset.subgraph and not CD_explain:
        #    batch_size = torch.max(batch) + 1
        #    x_list = []
        #    for i in range(batch_size):
        #        idx = (batch == i).nonzero()[0]
        #        #print(idx)
        #        x_list.append(x[idx])

        #    x = torch.cat(x_list, 0)
        if CD_explain:
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

    def loss(self, pred, label):
        #print(label)
        #print(pred)
        return F.nll_loss(pred, label)

class GCN_Node(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GCN_Node, self).__init__()
        self.dataset = dataset
        self.conv1 = CD_GCNConv(dataset.num_features, hidden, normalize = True, add_self_loops = True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(CD_GCNConv(hidden, hidden, normalize = True, add_self_loops = True))

        self.lin1 = CD_Linear(hidden, hidden)
        self.lin2 = CD_Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, CD_explain: bool = False, mask_index = None, Intermediate = False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if CD_explain:
            x = mask_x(x, mask_index)

        self.x0 = x
        self.edge_index = edge_index
        x = self.conv1(x, edge_index)
        self.x1 = x
        x = CD_relu(x)
        self.x2 = x
        self.x_convs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            self.x_convs.append(x)
            x = CD_relu(x)
            self.x_convs.append(x)
        self.x3 = x
        #x = CD_global_max_pool(x, batch)
        x = self.lin1(x)
        self.x4 = x
        x = CD_relu(x)
        self.x5 = x
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        self.x6 = x
        if CD_explain:
            self.x7 = x
            return x
        else:
            return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

    def loss(self, pred, label):
        return F.nll_loss(pred, label)



import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool, JumpingKnowledge
from CD_layers import CD_GCNConv, CD_global_max_pool, CD_global_mean_pool, CD_Linear, CD_relu, CD_GATConv



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



class ATTGCN_Node(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, num_features = None):
        super(ATTGCN_Node, self).__init__()
        num_features = dataset.num_features if num_features is None else num_features
        
        #self.conv1 = CD_GCNConv(num_features, hidden)
        self.conv1 = CD_GATConv(in_channels = num_features, out_channels = (int)(hidden/1), heads=1)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(CD_GATConv(in_channels = hidden, out_channels = (int)(hidden/1), heads=1))
            #self.convs.append(CD_GCNConv(hidden, hidden, normalize = True, add_self_loops = True))
        
        self.lin1 = CD_Linear(hidden, hidden, bias = True)
        
        self.lin2 = CD_Linear(hidden, dataset.num_classes, bias = True)
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, CD_explain: bool = False, mask_index = None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        self.middle_x = []

        if CD_explain:
            x = mask_x(x, mask_index)
        self.middle_x.append(x)

        x = CD_relu(self.conv1(x, edge_index))
        self.middle_x.append(x)
        for conv in self.convs:
            x = CD_relu(conv(x, edge_index))
            self.middle_x.append(x)
        #x = CD_global_max_pool(x, batch)
        #x = CD_feature_max_pool(x, batch)
        #print(x.shape)
        x = CD_relu(self.lin1(x))
        self.middle_x.append(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        self.middle_x.append(x)

        if CD_explain:
            return x
        else:
            return F.log_softmax(x, dim=-1)
            #return x
    def __repr__(self):
        return self.__class__.__name__

    def loss(self, pred, label):
        return F.nll_loss(pred, label)