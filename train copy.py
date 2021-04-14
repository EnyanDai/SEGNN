#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.ExplainGNN import ExplainGNN
from torch_geometric.datasets import Planetoid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=11, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--K', type=int, default=15,
                    help='trianing nodes')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')
parser.add_argument('--pred_K', type=int, default=5,
                    help='trianing nodes')
parser.add_argument('--hop', type=int, default=2,
                    help='hop')
parser.add_argument('--model', type=str, default='GCN',
                    choices=['MLP','GCN'])
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train.')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from dataset import get_labeled
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import WebKB

dataset = Planetoid(root='./data/',name='Cora')
data = dataset.data
from SynData import SynData
generator = SynData(data)
G = generator.syn_graph(n_basis=300,basis_type='real', noise_rate=0.0, connect=3)
data = generator.get_train_test_split(G).to(device)
#%%
features = data.x
idx = -1
score = features[idx] @ features.T


#%%
results = []
# for i in range(data.train_mask.shape[1]):

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# label_nodes = get_labeled(train_mask,data.edge_index,args.hop,device)
# val_nodes = get_labeled(val_mask,data.edge_index,args.hop,device)
# test_nodes = get_labeled(test_mask,data.edge_index,args.hop,device)

model = ExplainGNN(args,
            nfeat=data.x.shape[1],
            device=device).to(device)
model.fit(data.x, data.edge_index, data.edge_attr, data.y, label_nodes, val_nodes, train_iters=args.epochs, verbose=True)
# result = model.test(label_nodes,test_nodes)
# print(result)

from dataset import TestLoader
test_loader = TestLoader(test_mask, data.edge_index, sample_size=1, hop=args.hop, device=device)
acc = 0.0
explain_acc = 0.0
total_edge_nb = 0
for i in range(len(test_loader)):
    # print(data.y[test_loader[i][0]])
    node_acc, edge_acc, nb_edge = model.explain(data.x, data.edge_index, data.edge_attr, label_nodes, test_loader[i], G, K=1)
    acc += node_acc
    #print("node: {:4f}, edge: {:.4f}, {}".format(node_acc, edge_acc, nb_edge)) 
    explain_acc += nb_edge * edge_acc
    total_edge_nb += nb_edge

acc = acc/len(test_loader)
explain_acc = explain_acc/total_edge_nb
#all_acc.append(acc)
#all_explain.append(explain_acc)
print("Label similarity accuracy: {:.4f}".format(acc))
print("Edge similarity accuracy: {:.4f}".format(explain_acc))

#print(np.mean(results),np.std(results))
#print(np.mean(all_acc),np.std(all_acc))
#print(np.mean(all_explain),np.std(all_explain))
# %%
acc = 0.0
explain_acc = 0.0
total_edge_nb = 0
for i in range(len(test_loader)):
    # print(data.y[test_loader[i][0]])
    node_acc, edge_acc, nb_edge = model.basline(data.x, data.edge_index, data.edge_attr, label_nodes, test_loader[i], G, K=1)
    acc += node_acc
    #print("node: {:4f}, edge: {:.4f}, {}".format(node_acc, edge_acc, nb_edge)) 
    explain_acc += nb_edge * edge_acc
    total_edge_nb += nb_edge

acc = acc/len(test_loader)
explain_acc = explain_acc/total_edge_nb
#all_acc.append(acc)
#all_explain.append(explain_acc)
print("Label similarity accuracy: {:.4f}".format(acc))
print("Edge similarity accuracy: {:.4f}".format(explain_acc))