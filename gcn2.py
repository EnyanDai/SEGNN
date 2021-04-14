#%%
import os.path as osp

import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.datasets import WebKB
from dataset import get_syn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset = Planetoid(root='./data/',name='Cora')
dataset = WebKB(root='./data/',name='Cornell')
# data = get_syn().to(device)

data = dataset.data.to(device)
#%%
# data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.
from torch_geometric.utils import to_undirected
data.edge_index = to_undirected(data.edge_index)
data.adj_t = data.edge_index
data.y = data.y.long()

#%%
class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(data.x.shape[1], hidden_channels))
        self.lins.append(Linear(hidden_channels, int(data.y.max())+1))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=True))

        self.dropout = dropout

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)





import numpy as np
results = []
for i in range(data.train_mask.shape[1]):

    train_mask = data.train_mask[:,i]
    val_mask = data.val_mask[:,i]
    test_mask = data.test_mask[:,i]

    model = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
                shared_weights=True, dropout=0.6).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.convs.parameters(), weight_decay=0.01),
        dict(params=model.lins.parameters(), weight_decay=5e-4)
    ], lr=0.01)


    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.adj_t)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(data.x, data.adj_t).argmax(dim=-1)
        train_acc = int((pred[train_mask] == data.y[train_mask]).sum()) / int(train_mask.sum())
        val_acc = int((pred[val_mask] == data.y[val_mask]).sum()) / int(val_mask.sum())
        tmp_test_acc = int((pred[test_mask] == data.y[test_mask]).sum()) / int(test_mask.sum())
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:04d}, Loss: {loss:.4f} Train: {train_acc:.4f}, '
                f'Val: {val_acc:.4f}, Test: {tmp_test_acc:.4f}, '
                f'Final Test: {test_acc:.4f}')
    results.append(float(test_acc))
print(np.mean(results),np.std(results))