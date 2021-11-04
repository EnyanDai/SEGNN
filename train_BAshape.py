#%%
import argparse
import numpy as np
import torch
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
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--nlayer', type=int, default=2,
                    help='number of layers')
parser.add_argument('--K', type=int, default=10,
                    help='trianing nodes')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')
parser.add_argument('--beta1', type=float, default=1)
parser.add_argument('--beta2', type=float, default=1)
parser.add_argument('--T', type=float, default=1)
parser.add_argument('--sample_size', type=int, default=128,
                    help='sample_size')
parser.add_argument('--hop', type=int, default=1,
                    help='hop')
parser.add_argument('--attr_mask', type=float, default=0.5)
parser.add_argument('--model', type=str, default='DeGNN')
parser.add_argument('--init', action='store_true',
                    default=False, help='random initization the weight of aggregation or not')
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
from SynData import BA_shape 

data, edge_label_index = BA_shape('./data/syn1.pkl')
data = data.to(device)

train_mask = data.train_mask
val_mask = data.val_mask
test_mask = data.test_mask
np.random.seed(args.seed)
torch.manual_seed(args.seed)

from dataset import get_labeled, TestLoader, TrainLoader
label_nodes = get_labeled(train_mask, data.edge_index,args.hop,device)
train_loader = TrainLoader(train_mask, data.edge_index, sample_size=args.sample_size, hop=args.hop, device=device)
val_loader = TestLoader(val_mask, data.edge_index, int(val_mask.sum()), args.hop, device)

test_loader = TestLoader(test_mask, data.edge_index, sample_size=1, hop=2, device=device)

model = ExplainGNN(args,
            nfeat=data.x.shape[1],
            device=device).to(device)
model.fit(data.x, data.edge_index, data.edge_attr, data.y, label_nodes, train_loader, val_loader, train_iters=args.epochs, verbose=args.debug)


result, MAP = model.test(label_nodes, test_loader)
print("Accuracy: {:.4f}".format(result))

#%%
from utils import idx_to_mask
from sklearn.metrics import roc_auc_score
explain_mask = idx_to_mask(list(range(400,700,5)), len(train_mask))
explain_loader = TestLoader(explain_mask, data.edge_index, sample_size=1, hop=args.hop, device=device)

pred_all = []
real_all = []
for i in range(len(explain_loader)):
    edge_mask, edge_index = model.explain_structure(label_nodes, explain_loader[i],K=3)

    real = []
    for i in range(edge_index.shape[1]):
        real.append(edge_label_index[edge_index[0,i],edge_index[1,i]])
    real = np.asarray(real,dtype=np.float32)
    pred_all.append(edge_mask.cpu().numpy())
    real_all.append(real)

pred_all = np.concatenate((pred_all), axis=0)
real_all = np.concatenate((real_all), axis=0)

auc_all = roc_auc_score(real_all, pred_all)

print("The roc auc socre of crucial subgraph extraction on BA-Shape is {:.4f}".format(auc_all))

# %%
