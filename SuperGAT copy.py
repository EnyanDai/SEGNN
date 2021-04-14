import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SuperGATConv
from dataset import get_syn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = get_syn().to(device)
from torch_geometric.utils import to_undirected
data.edge_index = to_undirected(data.edge_index) 
data.y = data.y.long()
print(data.y.max())
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = SuperGATConv(data.x.shape[1], 8, heads=8,
                                  dropout=0.6, attention_type='MX',
                                  edge_sample_ratio=0.8, is_undirected=True)
        self.conv2 = SuperGATConv(8 * 8, int(data.y.max())+1, heads=8,
                                  concat=False, dropout=0.6,
                                  attention_type='MX', edge_sample_ratio=0.8,
                                  is_undirected=True)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        att_loss = self.conv1.get_attention_loss()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        att_loss += self.conv2.get_attention_loss()
        return F.log_softmax(x, dim=-1), att_loss


#%%
import numpy as np
results = []
for i in range(data.train_mask.shape[1]):

    train_mask = data.train_mask[:,i]
    val_mask = data.val_mask[:,i]
    test_mask = data.test_mask[:,i]

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out, att_loss = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], data.y[train_mask])
        loss += 4.0 * att_loss
        loss.backward()
        optimizer.step()
    
        model.eval()
        pred = model(data.x, data.edge_index)[0].argmax(dim=-1)
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