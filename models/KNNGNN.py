#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from models.GCN import GCN
import numpy as np
import torch_cluster
class Entropy(nn.Module):
    def __init__(self, eps=1e-8):
        super(Entropy, self).__init__()
        self.eps = 1e-8
    
    def forward(self,pred,target):
        eps = self.eps
        pred = pred.clamp(eps,1-(pred.shape[1]-1)*eps)
        loss = -torch.sum(target * torch.log(pred),dim=1)
        return loss.mean()


class KNNGNN(nn.Module):

    def __init__(self, args, nfeat, device=None):

        super(KNNGNN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nhid = args.hidden

        self.args = args

        self.GCN = GCN(self.nfeat,self.nhid,self.nhid, device=device)
        self.criterion = Entropy()

        self.best_model = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None


    def predict(self, features, edge_index, edge_weight, onehot_labels, idx_train, idx_test, train_phase=True):

        embedding = self.forward(features, edge_index, edge_weight)
        train_labels = onehot_labels[idx_train]
        score = embedding[idx_test] @ embedding[idx_train].T
        if train_phase:
            score.fill_diagonal_(float('-inf'))
        
        if self.args.K > 0:
            _,index = torch.topk(score, k=train_labels.shape[0]-self.args.K, dim=1, largest=False,sorted=False)
            score.scatter_(1,index,float('-inf'))

        weights = F.softmax(score,dim=1)
        preds = weights @ train_labels

        return preds



    def forward(self, x, edge_index, edge_weight):

        x = self.GCN(x,edge_index,edge_weight)

        return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False):
        """
        """
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        labels = labels.long()
        self.labels = labels
        self.onehot_labels = utils.tensor2onehot(labels)
        self.idx_train = idx_train
        self.GCN.initialize()

        self._train_with_val(idx_train, idx_val, train_iters, verbose)

    def _train_with_val(self, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        best_acc_val = 0
        

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.predict(self.features,self.edge_index,self.edge_weight,self.onehot_labels,idx_train,idx_train, train_phase=True)
            loss_train = F.mse_loss(output, self.onehot_labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.predict(self.features,self.edge_index,self.edge_weight,self.onehot_labels,idx_train,idx_val,train_phase=False)
            acc_val = utils.accuracy(output, self.labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if verbose and i+1 % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i+1, loss_train.item()),
                        'acc_val: {.4f}'.format(acc_val),
                        'best_acc_val'.format(best_acc_val))


        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict(self.features,self.edge_index,self.edge_weight,self.onehot_labels,self.idx_train,idx_test,train_phase=False)
        loss_test = self.criterion(output, self.onehot_labels[idx_test])
        acc_test = utils.accuracy(output, self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return output

# %%
