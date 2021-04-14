#%%
import time
import argparse
import numpy as np
import torch

from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.15, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train.')
parser.add_argument("--label_rate", type=float, default=0.1, help='rate of labeled data')
parser.add_argument("--r_type",type=str,default="add",
        choices=['add','remove','flip'])
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)

np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_train = idx_train[:int(args.label_rate * adj.shape[0])]

if args.attack == 'no':
    perturbed_adj = adj

if args.attack == 'random':
    from deeprobust.graph.global_attack import Random
    import random
    random.seed(15)


    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    attacker.attack(adj, n_perturbations, type=args.r_type)
    perturbed_adj = attacker.modified_adj

if args.attack == 'meta':
    perturbed_data = PrePtbDataset(root="./data".format(args.label_rate),
            name=args.dataset,
            attack_method=args.attack,
            ptb_rate=args.ptb_rate)
    perturbed_adj = perturbed_data.adj

if args.attack == 'nettack':
    import os
    import scipy.sparse as sp
    path = os.path.join("./data/{}".format(args.label_rate),"nettack/{}.npz".format(args.dataset))
    perturbed_adj = sp.load_npz(path)

#%%
from models.DoubleGCN import GCN
np.random.seed(args.seed)
torch.manual_seed(args.seed)

model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, device=device).to(device)

model.fit(features, perturbed_adj, labels, labels, idx_train, idx_val,train_iters=args.epochs)
output = model.test(idx_test)