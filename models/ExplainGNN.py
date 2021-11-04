#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from models.Backbone import *
import numpy as np

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
from torch_scatter import scatter

class Explainer(nn.Module):

    def __init__(self, nhid):
        super(Explainer, self).__init__()
    
    def explain(self, h, node_edge, label_edge):
        a = h
        edge_feature_n = (a[node_edge[0]] + a[node_edge[1]])/2
        edge_feature_l = (a[label_edge[0]] + a[label_edge[1]])/2
        edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)

        indices = edge_matrix.argmax(dim=1)
        
        return label_edge[:,indices]

    def explain_structure(self, h, node_edge, label_edge):
        
        a = h
        edge_feature_n = (a[node_edge[0]] + a[node_edge[1]])/2
        edge_feature_l = (a[label_edge[0]] + a[label_edge[1]])/2
        edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)

        score,_ = torch.max(edge_matrix, dim=1)


        return score



    def forward(self, h, nodes, label_nodes):
        a = h

        edge_feature_n = (a[nodes[3][0]] + a[nodes[3][1]])/2
        edge_feature_l = (a[label_nodes[3][0]] + a[label_nodes[3][1]])/2

        edge_matrix = -torch.cdist(edge_feature_n, edge_feature_l)


        edge_out_n = scatter(edge_matrix,label_nodes[4],dim=1,reduce='max')
        edge_out_n = scatter(edge_out_n,nodes[4], dim=0, reduce='mean')

        edge_out_l = scatter(edge_matrix,nodes[4], dim=0, reduce='max')
        edge_out_l = scatter(edge_out_l,label_nodes[4],dim=1,reduce='mean')
        
        edge_out = (edge_out_n + edge_out_l)/2
        return edge_out

from utils import attribute_mask
class ExplainGNN(nn.Module):

    def __init__(self, args, nfeat, device=None):

        super(ExplainGNN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nhid = args.hidden

        self.args = args

        if args.model=='GCN':
            self.model = GCN(self.nfeat,self.nhid)
        elif args.model=='MLP':
            self.model = MLP(self.nfeat,self.nhid)
        else:
            self.model = DeGNN(self.nfeat, self.nhid, args.nlayer, args.init)

        self.explainer = Explainer(self.nhid)
        self.criterion = Entropy()

        self.best_model = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def explain_structure(self, label_nodes, node, K=10):

        self.eval()
        embedding = self.embedding

        node_score = -torch.cdist(embedding[node[0]], embedding[label_nodes[0]])
        neigh_score = self.explainer(embedding,node,label_nodes)

        score = (self.args.alpha * node_score + (1-self.args.alpha) * neigh_score)/self.args.T

        _, indices = score.squeeze().topk(K)
        edge_mask = 0.0
        for i in range(K):

            label_edge = label_nodes[3][:,label_nodes[4]==indices[i]]
            edge_mask += self.explainer.explain_structure(self.embedding, node[3],label_edge)

        return edge_mask, node[3]


    def explain(self, label_nodes, node, G, K=1):
        
        self.eval()

        embedding = self.embedding

        node_score = -torch.cdist(embedding[node[0]], embedding[label_nodes[0]])
        neigh_score = self.explainer(embedding,node,label_nodes)

        score = (self.args.alpha * node_score + (1-self.args.alpha) * neigh_score)/self.args.T

        _, indices = score.squeeze().topk(K)
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

    def explain_top1(self, label_nodes, nodes):
        embedding = self.embedding

        node_score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])/self.args.T
        neigh_score = self.explainer(embedding,nodes,label_nodes)/self.args.T

        log_score = self.args.alpha * node_score + (1-self.args.alpha) * neigh_score

        _,index = torch.topk(log_score, k=1, dim=1)
        index = index[0,0]

        label_edge = label_nodes[3][:,label_nodes[4]==index]
        pair_edge = self.explainer.explain(self.embedding, nodes[3],label_edge)

        return label_nodes[0][index], label_edge, pair_edge

    def explain_rank(self, label_nodes, test_loader, role_id):
        self.eval()
        
        with torch.no_grad():
            embedding = self.forward(self.features, self.edge_index, self.edge_weight)
            self.embedding = embedding
            results = 0
            for i in range(len(test_loader)):
                nodes = test_loader[i]
                node_score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])/self.args.T
                neigh_score = self.explainer(embedding,nodes,label_nodes)/self.args.T

                log_score = self.args.alpha * node_score + (1-self.args.alpha) * neigh_score

                _,index = torch.sort(log_score, dim=1, descending=True)
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
    
    def contrast_loss(self, train_loader):

        # node score

        features_1 = attribute_mask(self.features, self.args.attr_mask)
        features_2 = attribute_mask(self.features, self.args.attr_mask)

        nodes, edges, neg_edges, pert_edge_index = train_loader.get_train()
        # print(features_2.shape, pert_edge_index.max(), pert_edge_index.min())
        embedding_1 = self.forward(features_1, self.edge_index, None)
        embedding_2 = self.forward(features_2, pert_edge_index, None)
  
       
        edge_feature_pos1 = (embedding_1[edges[0]] + embedding_1[edges[1]])/2
        edge_feature_pos2 = (embedding_2[edges[0]] + embedding_2[edges[1]])/2

        neg_row = embedding_2[neg_edges[0].flatten()]
        neg_col = embedding_2[neg_edges[1].flatten()]


        edge_feature_neg = (neg_row + neg_col)/2
        edge_feature_neg = torch.reshape(edge_feature_neg, (neg_edges.shape[1], neg_edges.shape[2],-1))

        edge_feature = torch.cat([edge_feature_pos2.unsqueeze(dim=0),edge_feature_neg], dim=0)
        edge_matrix = edge_feature - edge_feature_pos1
        edge_matrix = -torch.norm(edge_matrix, p=2, dim=-1).T/self.args.T

        edge_score = F.softmax(edge_matrix, dim=1)

        ground_truth = torch.zeros_like(edge_score)
        ground_truth[:,0] = 1.0
        edge_loss = self.criterion(edge_score, ground_truth)

        node_score = F.softmax(-torch.cdist(embedding_1[nodes], embedding_2[nodes])/self.args.T, dim=1)
        ground_truth = torch.eye(node_score.shape[0], device=self.device)

        node_loss = self.criterion(node_score, ground_truth)

        return self.args.beta1 * edge_loss + self.args.beta2 * node_loss
        



    def cls_loss(self, embedding, label_nodes):
        from torch_geometric.utils import softmax
        K = self.args.K
        Q = 15

        node_score = -torch.cdist(embedding[label_nodes[0]], embedding[label_nodes[0]])
        neigh_score = self.explainer(embedding,label_nodes,label_nodes)

        score = (self.args.alpha * node_score + (1-self.args.alpha) * neigh_score)/self.args.T

        _, indices = torch.topk(score, K, dim=1)

        pos_row = torch.arange(0, score.shape[0], device=self.device).long().repeat([K,1]).T.flatten()
        pos_col = indices.flatten()
        
        self_mask = (pos_col!=pos_row)
        pos_col = pos_col[self_mask]
        pos_row = pos_row[self_mask]

        # negative sample
        N = len(label_nodes[0])
        mask = torch.ones_like(score, dtype=torch.bool)
        mask.scatter_(1, indices, False)
        # train_labels = self.labels[label_nodes[0]]
        # repeat_label = train_labels.repeat([N,1])
        # mask = mask & ((repeat_label - repeat_label.T)!=0)
        mask = mask.view(-1)

        alpha = 2.0
        sample_size = min(int(N*Q*alpha), (N-K)*N)
        perm = torch.tensor(np.random.choice(N*N, sample_size, replace=False), device=self.device)
        perm = perm[mask[perm]][:N*Q]

        neg_row = perm // N
        neg_col = perm % N

        row = torch.cat([pos_row, neg_row])
        col = torch.cat([pos_col, neg_col])

        sample_score = score[row,col]
        label_score = softmax(sample_score, row)

        ref_labels = self.onehot_labels[label_nodes[0]][col].T
        preds = scatter(ref_labels * label_score, row).T

        cls_loss = self.criterion(preds, self.onehot_labels[label_nodes[0]])

        return cls_loss




    def predict(self, embedding, label_nodes, nodes):
        
        K = self.args.K

        node_score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])
        neigh_score = self.explainer(embedding,nodes,label_nodes)

        score = (self.args.alpha * node_score + (1-self.args.alpha) * neigh_score)/self.args.T
        
        if K > 0:
            _,index = torch.topk(score, k=score.shape[1]-K, dim=1, largest=False,sorted=False)
            score.scatter_(1,index,float('-inf'))
        score = F.softmax(score,dim=1)

        train_labels = self.onehot_labels[label_nodes[0]]
        preds = score @ train_labels
    
        return preds

    def forward(self, x, edge_index, edge_weight):
        x = self.model(x, edge_index)

        return x

    def fit(self, features, edge_index, edge_weight, labels, label_nodes, train_loader ,val_loader, train_iters=200, verbose=False):
        """
        """
        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features
        self.labels = labels
        self.onehot_labels = utils.tensor2onehot(labels)
        self.label_nodes = label_nodes
        self.idx_train = label_nodes[0]
        self._train_with_val(train_loader, val_loader, train_iters, verbose)

    
    def _train_with_val(self, train_loader, val_loader, train_iters, verbose):
        if verbose:
            print('=== training model ===')
        
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()

            self.embedding = self.forward(self.features, self.edge_index, self.edge_weight)
            cls_loss = self.cls_loss(self.embedding, self.label_nodes)

          
            # N = 1
            # cont_loss = 0
            # for j in range(N):
            cont_loss = self.contrast_loss(train_loader)
            # cont_loss = cont_loss/N
            loss_train = cls_loss + cont_loss

            loss_train.backward()
            optimizer.step()

            self.eval()

            self.embedding = self.forward(self.features, self.edge_index, self.edge_weight)
            acc_val = self.val(self.label_nodes,val_loader)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

            if verbose and (i+1) % 1 == 0:
                print('Epoch {}, cls loss: {}'.format(i+1, cls_loss.item()),
                        'cont_loss: {:.4f}'.format(cont_loss.item()),
                        'acc_val: {:.4f}'.format(acc_val),
                        'best_acc_val: {:.4f}'.format(best_acc_val))
        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def val(self, label_nodes, test_loader):
        self.eval()
        acc = 0
        n = 0
        self.embedding = self.forward(self.features, self.edge_index, self.edge_weight)
        for i in range(len(test_loader)):
            nodes = test_loader[i]
            preds = self.predict(self.embedding, label_nodes, nodes)
            acc += len(nodes[0]) * utils.accuracy(preds, self.labels[nodes[0]])
            n += len(nodes[0])
        acc = acc/n
        return float(acc)

    def test(self, label_nodes, test_loader):
        self.eval()
        acc = 0
        n = 0
        Pk = 0.0
        
        K = self.args.K
        with torch.no_grad():
            embedding = self.forward(self.features, self.edge_index, self.edge_weight)
            self.embedding = embedding
            for i in range(len(test_loader)):
                nodes = test_loader[i]
                node_score = -torch.cdist(embedding[nodes[0]], embedding[label_nodes[0]])/self.args.T
                neigh_score = self.explainer(embedding,nodes,label_nodes)/self.args.T

                log_score = self.args.alpha * node_score + (1-self.args.alpha) * neigh_score
        
                _,index = torch.topk(log_score, k=K, dim=1,sorted=False)
                index = torch.squeeze(index)
                score = log_score[:,torch.squeeze(index)]

                score = F.softmax(score,dim=1)

                train_labels = self.onehot_labels[label_nodes[0][index]]
                preds = score @ train_labels
                acc += len(nodes[0]) * utils.accuracy(preds, self.labels[nodes[0]])

                _,index = torch.topk(log_score, k=5, dim=1,sorted=False)
                index = torch.squeeze(index)

                y_true = (self.labels[label_nodes[0][index]]==self.labels[nodes[0]]).cpu().numpy()
                Pk += y_true.sum()/5

                n += 1

        acc = acc/n
        Pk = Pk/n
        return float(acc),Pk

# %%
