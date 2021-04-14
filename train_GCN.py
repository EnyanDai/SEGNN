#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
# from models.ExplainGNN import ExplainGNN
from torch_geometric.datasets import Planetoid, WebKB

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=11, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=500, help='Number of epochs to train.')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from dataset import get_labeled
from torch_geometric.utils import to_undirected
# from torch_geometric.datasets import WebKB
from torch_geometric.datasets import WikipediaNetwork
from dataset import get_syn
# dataset = Planetoid(root='./data/',name='Cora')
# dataset = WebKB(root='./data/',name='Cornell')
dataset = WikipediaNetwork('./data/',name='Chameleon')
#data = dataset.data.to(device)
# data = get_syn().to(device)

#%%

data.edge_index = to_undirected(data.edge_index)

results = []
for i in range(data.train_mask.shape[1]):

    train_mask = data.train_mask[:,i]
    val_mask = data.val_mask[:,i]
    test_mask = data.test_mask[:,i]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #%%
    model = GCN(nfeat=data.x.shape[1],\
                nhid=args.hidden,\
                nclass= int(data.y.max()+1),\
                dropout=args.dropout,\
                lr=args.lr,\
                weight_decay=args.weight_decay,\
                device=device).to(device)

    #%%
    from torch_geometric.utils import to_undirected
    data.edge_index = to_undirected(data.edge_index)
    model.fit(data.x, data.edge_index, data.edge_attr, data.y, train_mask, val_mask,train_iters=args.epochs)
    result = model.test(test_mask)
    print(result)
    results.append(result)

print(np.mean(results),np.std(results))

# %%
