#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from models.Backbone import *
import numpy as np
class Baseline(nn.Module):

    def __init__(self, args, nfeat, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4 ,device=None):

        super(Baseline, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.args = args
        nhid = self.args.hidden
        self.weight_decay = weight_decay

        self.body = self.get_body(nfeat, nhid, args.model)
        self.lin = nn.Linear(nhid, nclass)

        self.best_model = None
        self.edge_index = None
        self.embedding = None
        self.features = None
    
    def get_body(self, nfeat, nhid, model):
        if model=='GCN':
            return GCN(nfeat, nhid, dropout=self.dropout)
        if model=='GIN':
            return GIN(nfeat, nhid, dropout=self.dropout)
        if model=='MLP':
            return MLP(nfeat, nhid)
        return None

    def forward(self, x, edge_index):
        x = self.body(x, edge_index)
        y = self.lin(x)
        return y

    def predict(self, embedding, label_nodes, nodes):
        
        K = self.args.K

        embedding = self.embedding

        score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])/self.args.T
        
        if K > 0:
            _,index = torch.topk(score, k=score.shape[1]-K, dim=1, largest=False,sorted=False)
            score.scatter_(1,index,float('-inf'))
        score = F.softmax(score,dim=1)

        train_labels = self.onehot_labels[label_nodes[0]]
        preds = score @ train_labels
    
        return preds
    
    def explain(self, x, edge_index, label_nodes, node, G, K=1):

        self.eval()
        if self.embedding == None:
            self.embedding = self.body(x, edge_index)

        features = self.embedding

        score = -torch.cdist(features[node[0]], features[label_nodes[0]])
        _, indices = score.squeeze().topk(K)

        node_acc = 0.0
        edge_acc = 0.0
        for i in range(K):
            node_acc += (G.nodes[int(label_nodes[0][indices[i]])]['node_role']==G.nodes[int(node[0])]['node_role'])

            label_edge = label_nodes[3][:,label_nodes[4]==indices[i]]

            node_edge = node[3]
            edge_feature_n = (features[node_edge[0]] + features[node_edge[1]])/2
            
            edge_feature_l = (features[label_edge[0]] + features[label_edge[1]])/2
            edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)

            indices = edge_matrix.argmax(dim=1)
            pair_edge = label_edge[:,indices]
            edge_acc += self.check_edge_type(G, node[3], pair_edge)
        
        node_acc = node_acc/K
        edge_acc = edge_acc/K

        return float(node_acc), float(edge_acc), pair_edge.shape[1]

    def explain_top1(self, label_nodes, nodes):
        features = self.embedding

        score = -torch.cdist(features[nodes[0]], features[label_nodes[0]])

        _,index = torch.topk(score, k=1, dim=1,sorted=False)
        index = index[0]

        label_edge = label_nodes[3][:,label_nodes[4]==index]
        node_edge = nodes[3]
        edge_feature_n = (features[node_edge[0]] + features[node_edge[1]])/2
            
        edge_feature_l = (features[label_edge[0]] + features[label_edge[1]])/2
        edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)

        indices = edge_matrix.argmax(dim=1)
        pair_edge = label_edge[:,indices]

        return label_nodes[0][index], label_edge, pair_edge

    def explain_rank(self, label_nodes, test_loader, role_id):
        self.eval()
        
        with torch.no_grad():
            embedding = self.embedding
            results = 0
            for i in range(len(test_loader)):
                nodes = test_loader[i]
                node_score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])
                _,index = torch.sort(node_score, dim=1, descending=True)
                index = torch.squeeze(index)

                y_true = (role_id[label_nodes[0][index]]==role_id[nodes[0]]).cpu().numpy()
                Pk = []
                for k in range(1,30):
                    Pk.append(y_true[:k].sum()/k)
                Pk = np.asarray(Pk)

                results += Pk
            results = results/len(test_loader)
        
        return results


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
    
    def fit(self, features, edge_index, labels, idx_train, idx_val=None, train_iters=200, verbose=False):


        self.edge_index = edge_index
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.onehot_labels = utils.tensor2onehot(labels)
        self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.edge_index)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


    def test(self, idx_test, label_nodes, test_loader):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        self.embedding = self.body(self.features, self.edge_index)
        output = self.lin(self.embedding)
        acc_cls = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("cls test results:",
              "accuracy= {:.4f}".format(acc_cls.item()))
              
        acc = 0
        n = 0

        for i in range(len(test_loader)):
            nodes = test_loader[i]
            preds = self.predict(self.embedding, label_nodes, nodes)
            acc += len(nodes[0]) * utils.accuracy(preds, self.labels[nodes[0]])
            n += len(nodes[0])
        acc = acc/n
        return float(acc), float(acc_cls)
    
    def mAP(self, idx_test, label_nodes, test_loader):
        from sklearn.metrics import average_precision_score
        self.eval()
        MAP = 0.0
        n = 0
        with torch.no_grad():
            embedding = self.body(self.features, self.edge_index)
            for i in range(len(test_loader)):
                nodes = test_loader[i]
                log_score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])/self.args.T
                _,index = torch.topk(log_score, k=5, dim=1,sorted=False)
                index = torch.squeeze(index)
                y_true = (self.labels[label_nodes[0][index]]==self.labels[nodes[0]]).cpu().numpy()
                y_score = log_score[:,torch.squeeze(index)].squeeze().cpu().numpy()
   
                AP= average_precision_score(y_true,y_score)
                if np.isnan(AP):
                    AP = 0
                MAP += AP
                n += len(nodes[0])

        return MAP/n


    # def explain_rank(self, label_nodes, node, data):
        
    #     self.eval()
    #     embedding = self.embedding

    #     role_id = data.node_role[node[0]][0]
    #     label_node_idx = (data.node_role[label_nodes[0]]==role_id).cpu().nonzero().flatten().numpy()

    #     results = dict()

    #     for idx in label_node_idx:
            
    #         label_node = label_nodes[0][idx]
    #         label_edge = label_nodes[3][:, label_nodes[4]==idx]

    #         score = -torch.dist(embedding[node[0][0]],embedding[label_node])

    #         noise = data.noise[label_node]
    #         results[int(10*noise)]=score

    #     return results
# %%
