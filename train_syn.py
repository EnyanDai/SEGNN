#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.ExplainGNN import ExplainGNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--nlayer', type=int, default=2,
                    help='number of layers')
parser.add_argument('--K', type=int, default=5,
                    help='trianing nodes')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')
parser.add_argument('--beta1', type=float, default=1)
parser.add_argument('--beta2', type=float, default=1)
parser.add_argument('--T', type=float, default=1)
parser.add_argument('--sample_size', type=int, default=256,
                    help='sample_size')
parser.add_argument('--hop', type=int, default=2,
                    help='hop')
parser.add_argument('--model', type=str, default='DeGNN')
parser.add_argument('--init', action='store_true',
                    default=False, help='random initization the weight of aggregation or not')
parser.add_argument('--attr_mask', type=float, default=0.2)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=50, help='Number of epochs to train.')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from dataset import get_labeled
import networkx as nx
from torch_geometric.utils import from_networkx
G = nx.read_gpickle("./data/Syn-Cora.gpickle")
data = from_networkx(G).to(device)
#%%
train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask
np.random.seed(args.seed)
torch.manual_seed(args.seed)

from dataset import get_labeled, TestLoader, TrainLoader
label_nodes = get_labeled(train_mask, data.edge_index,args.hop,device)
train_loader = TrainLoader(train_mask, data.edge_index, sample_size=128, hop=args.hop, device=device)
val_loader = TestLoader(val_mask, data.edge_index, int(val_mask.sum()), args.hop, device)

test_loader = TestLoader(test_mask, data.edge_index, sample_size=1, hop=args.hop, device=device)

#%%
model = ExplainGNN(args,
            nfeat=data.x.shape[1],
            device=device).to(device)
model.fit(data.x, data.edge_index, data.edge_attr, data.y, label_nodes, train_loader, val_loader, train_iters=args.epochs, verbose=args.debug)


result, MAP = model.test(label_nodes, test_loader)
print("Accuracy: {:.4f}".format(result))

acc = 0.0
explain_acc = 0.0
total_edge_nb = 0
for i in range(len(test_loader)):
    node_acc, edge_acc, nb_edge = model.explain(label_nodes, test_loader[i], G, K=1)
    acc += node_acc
    explain_acc += nb_edge * edge_acc
    total_edge_nb += nb_edge

acc = acc/len(test_loader)
explain_acc = explain_acc/total_edge_nb

print("Node Pair accuracy: {:.4f}".format(acc))
print("Edge Pair accuracy: {:.4f}".format(explain_acc))
# %%
