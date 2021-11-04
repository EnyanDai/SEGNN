#%%
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--K', type=int, default=10,
                    help='trianing nodes')
parser.add_argument('--T', type=float, default=1)
parser.add_argument('--sample_size', type=int, default=256,
                    help='sample_size')
parser.add_argument('--hop', type=int, default=2,
                    help='hop')
parser.add_argument('--model', type=str, default='MLP',
                    choices=['MLP','GCN','GIN'])
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=100, help='Number of epochs to train.')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from dataset import get_labeled


#%%
results = []
# for i in range(data.train_mask.shape[1]):
from dataset import get_labeled
import networkx as nx
from torch_geometric.utils import from_networkx
G = nx.read_gpickle("./data/Syn-Cora.gpickle")
data = from_networkx(G).to(device)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask
np.random.seed(args.seed)
torch.manual_seed(args.seed)

from dataset import get_labeled, TestLoader, TrainLoader
label_nodes = get_labeled(train_mask, data.edge_index,args.hop,device)
test_loader = TestLoader(test_mask, data.edge_index, sample_size=1, hop=args.hop, device=device)

from models.Baseline import Baseline
model = Baseline(args,\
            nfeat=data.x.shape[1],\
            nclass= int(data.y.max()+1),\
            dropout=args.dropout,\
            lr=args.lr,\
            weight_decay=args.weight_decay,\
            device=device).to(device)
model.fit(data.x, data.edge_index, data.y, train_mask, val_mask,train_iters=args.epochs)

acc_k, acc_cls= model.test(test_mask, label_nodes, test_loader)
print("Accuracy of KNN: {:.4f}, Accuracy of CLS: {:.4f}".format(acc_k, acc_cls))

acc = 0.0
explain_acc = 0.0
total_edge_nb = 0
for i in range(len(test_loader)):
    node_acc, edge_acc, nb_edge = model.explain(data.x, data.edge_index, label_nodes, test_loader[i], G, K=1)
    acc += node_acc
    explain_acc += nb_edge * edge_acc
    total_edge_nb += nb_edge

acc = acc/len(test_loader)
explain_acc = explain_acc/total_edge_nb
print("Baseline Node Pair accuracy: {:.4f}".format(acc))
print("Baseline Edge Pair accuracy: {:.4f}".format(explain_acc))
import os
torch.save(model.state_dict(), os.path.join('checkpoint', "{}_K_syn".format(args.model)))