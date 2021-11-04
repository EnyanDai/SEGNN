#%%
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from dataset import Planetoid
from deeprobust.graph.data import Dataset

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
# K = 80
parser.add_argument('--K', type=int, default=40,
                    help='trianing nodes')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')

parser.add_argument('--T', type=float, default=1,
                    help='Temp')

parser.add_argument('--hop', type=int, default=2,
                    help='hop')
parser.add_argument('--model', type=str, default='MLP',
                    choices=['MLP','GCN','GIN'])
parser.add_argument('--dropout', type=float, default=0.15,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train.')

parser.add_argument('--dataset', type=str, default='Cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)
import torch_geometric.transforms as T
if args.dataset in ["Cora","Pubmed"]:
    dataset = Planetoid('./data/', args.dataset)
    data = dataset[0].to(device)
if args.dataset=="citeseer":
    np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # idx_train = idx_train[:int(args.label_rate * adj.shape[0])]

    from torch_geometric.utils import from_scipy_sparse_matrix
    from torch_geometric.data import Data
    from utils import idx_to_mask

    edge_index, edge_weight = from_scipy_sparse_matrix(adj)
    features = torch.FloatTensor(features.toarray())
    labels = torch.LongTensor(labels)
    train_mask = idx_to_mask(idx_train, features.shape[0])
    test_mask = idx_to_mask(idx_test, features.shape[0])
    val_mask = idx_to_mask(idx_val, features.shape[0])

    data = Data(x=features, y=labels, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    data = data.to(device)

#%%
train_mask = data.train_mask

val_mask = data.val_mask
test_mask = data.test_mask
seed=args.seed
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
import random
random.seed(seed)

#%%
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
model.fit(data.x, data.edge_index, data.y, train_mask, val_mask,train_iters=args.epochs,verbose=True)

acc_k, acc_cls= model.test(test_mask, label_nodes, test_loader)
mAP = model.mAP(test_mask, label_nodes, test_loader)
print("Accuracy of KNN: {:.4f}, Accuracy of CLS: {:.4f}, MAP: {:.4f}".format(acc_k, acc_cls, mAP))
import os
torch.save(model.state_dict(), os.path.join('checkpoint', "{}_K_{}".format(args.model,args.dataset)))
# %%
