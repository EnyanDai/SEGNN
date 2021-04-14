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

def cosine(t1, t2):

    return t1@t2.T/(t1.norm(dim=1, keepdim=True)@t2.norm(dim=1, keepdim=True).T)

#%%
from torch_scatter import scatter,scatter_softmax

class Explainer(nn.Module):

    def __init__(self, nhid):
        super(Explainer, self).__init__()
        self.linear_att = torch.nn.Linear(nhid,nhid)
        self.linear_metric = torch.nn.Linear(nhid,nhid)
        self.edge_linear = torch.nn.Linear(2*nhid,nhid)
    
    def explain(self, h, node_edge, label_edge):
        a = h
        # edge score
        #edge_feature_n = self.edge_linear(torch.cat([a[node_edge[0]],a[node_edge[1]]], dim=1))
        edge_feature_n = (a[node_edge[0]] + a[node_edge[1]])/2
        #edge_feature_l = self.edge_linear(torch.cat([a[label_edge[0]],a[label_edge[1]]], dim=1))
        edge_feature_l = (a[label_edge[0]] + a[label_edge[1]])/2
        edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)

        indices = edge_matrix.argmax(dim=1)
        
        return label_edge[:,indices]


    def forward(self, h, nodes, label_nodes):
        a = h
        # node score
        # matrix = a[nodes[1]] @ h[label_nodes[1]].T
        # node_out = scatter(matrix,label_nodes[2],dim=1,reduce='max')
        # node_out = scatter(node_out,nodes[2], dim=0, reduce='mean')
        # edge score
        #edge_feature_n = self.edge_linear(torch.cat([a[nodes[3][0]],a[nodes[3][1]]], dim=1))
        edge_feature_n = (a[nodes[3][0]] + a[nodes[3][1]])/2
        #edge_feature_l = self.edge_linear(torch.cat([a[label_nodes[3][0]],a[label_nodes[3][1]]], dim=1))
        edge_feature_l = (a[label_nodes[3][0]] + a[label_nodes[3][1]])/2
        edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)


        edge_out_n = scatter(edge_matrix,label_nodes[4],dim=1,reduce='max')
        edge_out_n = scatter(edge_out_n,nodes[4], dim=0, reduce='mean')

        edge_out_l = scatter(edge_matrix,nodes[4], dim=0, reduce='max')
        edge_out_l = scatter(edge_out_l,label_nodes[4],dim=1,reduce='mean')
        
        edge_out = (edge_out_n + edge_out_l)/2
        return edge_out

from models.MLP import MLP

class ExplainGNN(nn.Module):

    def __init__(self, args, nfeat, device=None):

        super(ExplainGNN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nhid = args.hidden

        self.args = args

        if args.model=='GCN':
            self.model = GCN(self.nfeat,self.nhid,self.nhid, device=device)
        else:
            self.model = MLP(self.nfeat,self.nhid,self.nhid, device=device)
        
        self.explainer = Explainer(self.nhid)
        self.criterion = Entropy()

        self.best_model = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
    
    def basline(self, features, edge_index, edge_weight, label_nodes, node, G, K=1):

        score = -torch.cdist(features[node[0]], features[label_nodes[0]])
        _, indices = score.squeeze().topk(K)
        # indices = label_nodes[0][indices]
        node_acc = 0.0
        edge_acc = 0.0
        for i in range(K):
            node_acc += (G.nodes[int(label_nodes[0][indices[i]])]['node_role']==G.nodes[int(node[0])]['node_role'])

            label_edge = label_nodes[3][:,label_nodes[4]==indices[i]]

            node_edge = node[3]
            edge_feature_n = (features[node_edge[0]] + features[node_edge[1]])/2
            #edge_feature_l = self.edge_linear(torch.cat([a[label_edge[0]],a[label_edge[1]]], dim=1))
            edge_feature_l = (features[label_edge[0]] + features[label_edge[1]])/2
            edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)

            indices = edge_matrix.argmax(dim=1)
            pair_edge = label_edge[:,indices]
            edge_acc += self.check_edge_type(G, node[3], pair_edge)
        
        node_acc = node_acc/K
        edge_acc = edge_acc/K

        return float(node_acc), float(edge_acc), pair_edge.shape[1]

    def explain(self, features, edge_index, edge_weight, label_nodes, node, G, K=1):
        
        self.eval()
        score = self.predict(features, edge_index, edge_weight, label_nodes, node, train_phase=False)
        _, indices = score.squeeze().topk(K)
        # indices = label_nodes[0][indices]
        #print(indices)
        #print(node[0])
        node_acc = 0.0
        edge_acc = 0.0
        for i in range(K):
            node_acc += (G.nodes[int(label_nodes[0][indices[i]])]['node_role']==G.nodes[int(node[0])]['node_role'])

            label_edge = label_nodes[3][:,label_nodes[4]==indices[i]]
            pair_edge = self.explainer.explain(self.embedding, node[3],label_edge)
            edge_acc += self.check_edge_type(G, node[3], pair_edge)
        
        node_acc = node_acc/K
        edge_acc = edge_acc/K

        return float(node_acc), float(edge_acc), pair_edge.shape[1]

    def check_edge_type(self, G, node_edge, pair_edge):
        # to do
        n = 0
        t = 0
        for i in range(node_edge.shape[1]):
            edge_role0 = G.edges[int(node_edge[0, i]), int(node_edge[1, i])]['edge_role']

            edge_role1 = G.edges[int(pair_edge[0, i]), int(pair_edge[1, i])]['edge_role']
            if edge_role0 >0 :
                n += 1
                if edge_role0==edge_role1:
                    t +=1
        if n==0:
            return 1.0

        return t/n



    def predict(self, features, edge_index, edge_weight, label_nodes, nodes, train_phase, K=None):
        if K==None:
            K = self.args.K

        embedding = self.forward(features, edge_index, edge_weight)
        self.embedding = embedding

        node_score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])
        if train_phase:
            node_score.fill_diagonal_(float('-inf'))
        
        if K > 0:
            _,index = torch.topk(node_score, k=node_score.shape[1]-K, dim=1, largest=False,sorted=False)
            node_score.scatter_(1,index,float('-inf'))

        # score = score.T.div(score.sum(dim=1)).T
        node_score = F.softmax(node_score,dim=1)
        
        neigh_score = self.explainer(embedding,nodes,label_nodes)

        if train_phase:
            neigh_score.fill_diagonal_(float('-inf'))
        
        if K > 0:
            _,index = torch.topk(neigh_score, k=neigh_score.shape[1]-K, dim=1, largest=False,sorted=False)
            neigh_score.scatter_(1,index,float('-inf'))

        # score = score.T.div(score.sum(dim=1)).T
        neigh_score = F.softmax(neigh_score,dim=1)
        # print(neigh_score.shape)
        score = self.args.alpha * node_score + (1-self.args.alpha) * neigh_score
        return score

    def forward(self, x, edge_index, edge_weight):
        if self.args.model=='GCN':
            x = self.model(x,edge_index,edge_weight)
        else:
            x = self.model(x)

        return x

    def fit(self, features, edge_index, edge_weight, labels, train_mask, val_mask, train_iters=200, verbose=False):
        """
        """
        from dataset import get_labeled, TestLoader
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        self.labels = labels
        self.onehot_labels = utils.tensor2onehot(labels)
        self.label_nodes = get_labeled(train_mask,edge_index,self.args.hop,self.device)
        self.val_loader = TestLoader(val_mask, edge_index, self.args.sample_size, self.args.hop, self.device)
        
        self.idx_train = self.label_nodes[0]

        self._train_with_val(self.label_nodes, self.val_loader, train_iters, verbose)

    def _train_with_val(self, label_nodes, val_loader, train_iters, verbose):
        if verbose:
            print('=== training model ===')
        
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            score = self.predict(self.features,self.edge_index,self.edge_weight,label_nodes,label_nodes,train_phase=True)
            train_labels = self.onehot_labels[self.idx_train]
            preds = score @ train_labels
            
            loss_train = self.criterion(preds, train_labels)
            loss_train.backward()
            optimizer.step()

            self.eval()

            acc_val = self.test(label_nodes,val_loader)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

            if verbose and (i+1) % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i+1, loss_train.item()),
                        'acc_val: {:.4f}'.format(acc_val),
                        'best_acc_val: {:.4f}'.format(best_acc_val))
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, label_nodes, test_loader):

        self.eval()
        acc = 0
        for i in range(len(test_loader)):
            nodes = test_loader[i]
            score = self.predict(self.features,self.edge_index,self.edge_weight,label_nodes, nodes, train_phase=False, K=self.args.pred_K)
            train_labels = self.onehot_labels[label_nodes[0]]
            preds = score @ train_labels
            acc += utils.accuracy(preds, self.labels[nodes[0]])
        acc = acc/len(test_loader)
        return float(acc)

# %%


# G = nx.Graph()
# # G.add
# %%


# %%
