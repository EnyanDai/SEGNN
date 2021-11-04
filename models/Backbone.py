#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class DeConv(MessagePassing):
    def __init__(self, in_channels, init,**kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(DeConv, self).__init__(**kwargs)

        self.in_channels = in_channels

        self.lin_l = torch.nn.Parameter(0.01*torch.ones([in_channels]))
        if init:
            torch.nn.init.uniform_(self.lin_l, 0.01) 

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = out * self.lin_l

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels)

class DeGNN(nn.Module):

    def __init__(self, nfeat, nhid, nlayer, init=False):

        super(DeGNN, self).__init__()


        body = [nn.Linear(nfeat,nhid)]
        for l in range(nlayer-1):
            body.append(nn.ReLU())
            body.append(nn.Linear(nhid,nhid))
        
        self.body = nn.Sequential(*body)
        self.agg = DeConv(nhid, init)

    def forward(self, x, edge_index):
        h = self.body(x)
        neigh = self.agg(h, edge_index)
        return h + neigh

# %%
class GCN(nn.Module):

    def __init__(self, nfeat, nhid, dropout=0.5):

        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index)
        return x

class MLP(nn.Module):

    def __init__(self, nfeat, nhid):

        super(MLP, self).__init__()

        self.body = nn.Sequential(nn.Linear(nfeat,nhid),
                                    nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(nhid,nhid))

    def forward(self, x, edge_index):
        x = self.body(x)
        return x

class GIN(nn.Module):

    def __init__(self, nfeat, nhid, dropout=0.5):

        super(GIN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.lin_1 = nn.Linear(nhid, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.lin_2 = nn.Linear(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.lin_1(F.relu(self.gc1(x, edge_index)))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin_2(F.relu(self.gc2(x, edge_index)))

        return x
