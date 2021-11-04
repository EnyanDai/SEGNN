#%%
import time
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from dataset import Dataset
from torch_geometric.datasets import Planetoid
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=13, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
# K = 80
parser.add_argument('--K', type=int, default=20,
                    help='trianing nodes')
parser.add_argument('--init', action='store_true',
                    default=False, help='random initization the weight of aggregation or not')
parser.add_argument('--nlayer', type=int, default=3,
                    help='number of layers')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')
parser.add_argument('--beta1', type=float, default=0.01, help='edge')
parser.add_argument('--beta2', type=float, default=0.01, help='node')
parser.add_argument('--T', type=float, default=1,
                    help='Temp')
parser.add_argument('--hop', type=int, default=2,
                    help='hop')
parser.add_argument('--model', type=str, default='DeGNN',
                    choices=['MLP','GCN','DeGNN'])

parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train.')
parser.add_argument('--attr_mask', type=float, default=0.15)
parser.add_argument('--dataset', type=str, default='Pubmed', help='dataset')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

print(args)
seed = args.seed
# for seed in range(10,15):
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
import random
random.seed(seed)
import torch_geometric.transforms as T
if args.dataset in ["Cora","Pubmed"]:
    dataset = Planetoid('./data/', args.dataset)
    data = dataset[0].to(device)
if args.dataset=="citeseer":
    np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

    data = Dataset(root='./data', name=args.dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

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


from models.ExplainGNN import ExplainGNN
train_mask = data.train_mask

val_mask = data.val_mask
test_mask = data.test_mask

from dataset import get_labeled, TestLoader, TrainLoader
label_nodes = get_labeled(train_mask, data.edge_index,args.hop,device)
train_loader = TrainLoader(train_mask, data.edge_index, sample_size=args.batch_size, hop=args.hop, device=device)
val_loader = TestLoader(val_mask, data.edge_index, 16, args.hop, device)

#%%
model = ExplainGNN(args,
            nfeat=data.x.shape[1],
            device=device).to(device)
model.fit(data.x, data.edge_index, data.edge_attr, data.y, label_nodes, train_loader,val_loader, train_iters=args.epochs, verbose=args.debug)

from dataset import TestLoader
test_loader = TestLoader(test_mask, data.edge_index, sample_size=1, hop=args.hop, device=device)

ACC, mAP= model.test(label_nodes, test_loader)
print("Accuracy: {:.4f}, mAP: {:.4f}".format(ACC, mAP))

torch.save(model.state_dict(),"checkpoint/SEGNN_{}_{:.3f}".format(args.dataset, ACC))
# %%
