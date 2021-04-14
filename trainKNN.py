#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
from models.KNNGNN import KNNGNN
from torch_geometric.datasets import Planetoid,WebKB

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
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--K', type=int, default=-1,
                    help='trianing nodes')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
# dataset = Planetoid(root='./data/',name='Cora')
dataset = WebKB(root='./data/',name='Wisconsin')
data = dataset.data.to(device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

model = KNNGNN(args,
            nfeat=data.x.shape[1],
            device=device).to(device)

model.fit(data.x, data.edge_index, data.edge_attr, data.y, data.train_mask[:,0], data.val_mask[:,0],train_iters=args.epochs)
output = model.test(data.test_mask[:,0])

# %%