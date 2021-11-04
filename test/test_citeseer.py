#%%
import time
import argparse
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
from deeprobust.graph.data import Dataset, PrePtbDataset
from torch_geometric.datasets import Planetoid
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
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
# K = 80
parser.add_argument('--K', type=int, default=40,
                    help='trianing nodes')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')
parser.add_argument('--beta', type=float, default=0.01)
parser.add_argument('--T', type=float, default=1,
                    help='Temp')
parser.add_argument('--hop', type=int, default=2,
                    help='hop')
parser.add_argument('--model', type=str, default='DeGNN',
                    choices=['MLP','GCN','DeGNN'])
parser.add_argument('--init', action='store_true',
                    default=False, help='random initization the weight of aggregation or not')
parser.add_argument('--nlayer', type=int, default=2,
                    help='number of layers')
parser.add_argument('--epochs', type=int,  default=80, help='Number of epochs to train.')
parser.add_argument('--attr_mask', type=float, default=0.4)
parser.add_argument('--dataset', type=str, default='citeseer', help='dataset')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)
seed = args.seed
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

    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
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

from models.ExplainGNN import ExplainGNN
train_mask = data.train_mask

val_mask = data.val_mask
test_mask = data.test_mask

from dataset import get_labeled, TestLoader, TrainLoader
label_nodes = get_labeled(train_mask, data.edge_index,args.hop,device)
train_loader = TrainLoader(train_mask, data.edge_index, sample_size=args.batch_size, hop=args.hop, device=device)
val_loader = TestLoader(val_mask, data.edge_index, 16, args.hop, device)

#%%
import utils
model_EX = ExplainGNN(args,
            nfeat=data.x.shape[1],
            device=device).to(device)
model_EX.features = data.x
model_EX.labels = data.y
model_EX.onehot_labels = utils.tensor2onehot(data.y)
model_EX.edge_index = data.edge_index
model_EX.load_state_dict(torch.load("checkpoint/ExGNN_citeseer_0.749"))

from dataset import TestLoader
test_loader = TestLoader(test_mask, data.edge_index, sample_size=1, hop=args.hop, device=device)
ACC, mAP= model_EX.test(label_nodes, test_loader)
precision_k = model_EX.explain_rank(label_nodes, test_loader, data.y)
print("Accuracy: {:.4f}, mAP: {:.4f}".format(ACC, mAP))

#%%

args.model = 'GCN'
from models.Baseline import Baseline
model_gcn_k = Baseline(args,\
            nfeat=data.x.shape[1],\
            nclass= int(data.y.max()+1),\
            dropout=0.5,\
            lr=args.lr,\
            weight_decay=args.weight_decay,\
            device=device).to(device)

model_gcn_k.features = data.x
model_gcn_k.labels = data.y
model_gcn_k.onehot_labels = utils.tensor2onehot(data.y)
model_gcn_k.edge_index = data.edge_index
model_gcn_k.load_state_dict(torch.load("checkpoint/{}_K_{}".format(args.model,args.dataset)))

acc_k, acc_cls= model_gcn_k.test(test_mask, label_nodes, test_loader)
mAP = model_gcn_k.mAP(test_mask, label_nodes, test_loader)
precision_k_gcn = model_gcn_k.explain_rank(label_nodes, test_loader, data.y)
print("Accuracy of KNN: {:.4f}, Accuracy of CLS: {:.4f}, MAP: {:.4f}".format(acc_k, acc_cls, mAP))

#%%
args.model = 'GIN'
from models.Baseline import Baseline
model_GIN_k = Baseline(args,\
            nfeat=data.x.shape[1],\
            nclass= int(data.y.max()+1),\
            dropout=0.5,\
            lr=args.lr,\
            weight_decay=args.weight_decay,\
            device=device).to(device)

model_GIN_k.features = data.x
model_GIN_k.labels = data.y
model_GIN_k.onehot_labels = utils.tensor2onehot(data.y)
model_GIN_k.edge_index = data.edge_index
model_GIN_k.load_state_dict(torch.load("checkpoint/{}_K_{}".format(args.model,args.dataset)))

acc_k, acc_cls= model_GIN_k.test(test_mask, label_nodes, test_loader)
mAP = model_GIN_k.mAP(test_mask, label_nodes, test_loader)
print("Accuracy of KNN: {:.4f}, Accuracy of CLS: {:.4f}, MAP: {:.4f}".format(acc_k, acc_cls, mAP))
precision_k_gin = model_GIN_k.explain_rank(label_nodes, test_loader, data.y)

args.model = 'MLP'
from models.Baseline import Baseline
model_mlp_k = Baseline(args,\
            nfeat=data.x.shape[1],\
            nclass= int(data.y.max()+1),\
            dropout=0.5,\
            lr=args.lr,\
            weight_decay=args.weight_decay,\
            device=device).to(device)

model_mlp_k.features = data.x
model_mlp_k.labels = data.y
model_mlp_k.onehot_labels = utils.tensor2onehot(data.y)
model_mlp_k.edge_index = data.edge_index
model_mlp_k.load_state_dict(torch.load("checkpoint/{}_K_{}".format(args.model,args.dataset)))

acc_k, acc_cls= model_mlp_k.test(test_mask, label_nodes, test_loader)
mAP = model_mlp_k.mAP(test_mask, label_nodes, test_loader)
print("Accuracy of KNN: {:.4f}, Accuracy of CLS: {:.4f}, MAP: {:.4f}".format(acc_k, acc_cls, mAP))
precision_k_mlp = model_mlp_k.explain_rank(label_nodes, test_loader, data.y)

#%%
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 30, 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(9,8))
k = 8
x = np.arange(1,k+1)

plt.plot(x,100*precision_k[:k],label="Ours",linestyle="-",linewidth=4,markersize=12,marker="o")
plt.plot(x,100*precision_k_gcn[:k],label="GCN-K",linestyle="-",linewidth=4,markersize=12,marker="^")
plt.plot(x,100*precision_k_gin[:k],label="GIN-K",linestyle="-",linewidth=4,markersize=12,marker="s")
plt.plot(x,100*precision_k_mlp[:k],label="MLP-K",linestyle="-",linewidth=4,markersize=12,marker="D")


plt.xlabel("k",fontsize=40)
plt.ylabel("Precision@k (%)",fontsize=40)
# plt.xlim(-1,26)
plt.ylim(61,73)
plt.xticks([2,4,6,8],fontsize=35)
plt.yticks(fontsize=35)
plt.legend(fontsize=22,ncol=2,loc="upper right")
plt.tight_layout()
plt.savefig("precision_citeseer.pdf")

# %%
