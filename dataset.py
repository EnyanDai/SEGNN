#%%
import torch
from torch_geometric.utils import k_hop_subgraph,is_undirected, num_nodes


import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import pickle as pkl
import networkx as nx
from utils import get_train_val_test

class Dataset():
    """Dataset class contains four citation network datasets "cora", "cora-ml", "citeseer" and "pubmed",
    and one blog dataset "Polblogs".
    The 'cora', 'cora-ml', 'poblogs' and 'citeseer' are downloaded from https://github.com/danielzuegner/gnn-meta-attack/tree/master/data, and 'pubmed' is from https://github.com/tkipf/gcn/tree/master/gcn/data.

    Parameters
    ----------
    root :
        root directory where the dataset should be saved.
    name :
        dataset name, it can be choosen from ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed']
    seed :
        random seed for splitting training/validation/test.
    --------
	We can first create an instance of the Dataset class and then take out its attributes.

	>>> from deeprobust.graph.data import Dataset
	>>> data = Dataset(root='/tmp/', name='cora')
	>>> adj, features, labels = data.adj, data.features, data.labels
	>>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    """

    def __init__(self, root, name, seed=None):
        self.name = name.lower()

        assert self.name in ['cora', 'citeseer', 'cora_ml', 'polblogs', 'pubmed'], \
            'Currently only support cora, citeseer, cora_ml, polblogs, pubmed'

        self.seed = seed
        
        self.url =  'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/%s.npz' % self.name
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'

        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()

    def get_train_val_test(self):
        """Get training, validation, test splits
        """
        return get_train_val_test(nnodes=self.adj.shape[0], val_size=0.1, test_size=0.8, stratify=self.labels, seed=self.seed)

    def load_data(self):
        print('Loading {} dataset...'.format(self.name))
        if self.name == 'pubmed':
            return self.load_pubmed()

        if not osp.exists(self.data_filename):
            self.download_npz()

        adj, features, labels = self.get_adj()
        return adj, features, labels

    def download_npz(self):
        """Download adjacen matrix npz file from self.url.
        """
        print('Dowloading from {} to {}'.format(self.url, self.data_filename))
        try:
            urllib.request.urlretrieve(self.url, self.data_filename)
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')

    def download_pubmed(self, name):
        url = 'https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data/'
        try:
            urllib.request.urlretrieve(url + name, osp.join(self.root, name))
        except:
            raise Exception('''Download failed! Make sure you have stable Internet connection and enter the right name''')


    def load_pubmed(self):
        dataset = 'pubmed'
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            name = "ind.{}.{}".format(dataset, names[i])
            data_filename = osp.join(self.root, name)

            if not osp.exists(data_filename):
                self.download_pubmed(name)

            with open(data_filename, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)


        test_idx_file = "ind.{}.test.index".format(dataset)
        if not osp.exists(osp.join(self.root, test_idx_file)):
            self.download_pubmed(test_idx_file)

        test_idx_reorder = parse_index_file(osp.join(self.root, test_idx_file))
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = np.where(labels)[1]
        return adj, features, labels

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        lcc = self.largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        features = features[lcc]
        labels = labels[lcc]
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels

    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            # loader = dict(loader)
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def largest_connected_components(self, adj, n_components=1):
        """Select k largest connected components.

		Parameters
		----------
		adj : scipy.sparse.csr_matrix
			input adjacency matrix
		n_components : int
			n largest connected components we want to select
		"""

        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels]
        return onehot_mx

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def get_batch(idx, edge_index, hop, device):
        nodes = torch.tensor(idx, dtype=torch.long, device=device)

        all_neighbors = []
        all_subgraphs = []
        batch_n = []
        batch_g = []
        undirected = is_undirected(edge_index)
        for i,node in enumerate(nodes):
            neighbors, subgraphs,_,_=k_hop_subgraph(int(node),hop,edge_index)
            # if undirected:
            subgraphs = subgraphs[:,subgraphs[1]>=subgraphs[0]]
            all_neighbors.append(neighbors)
            all_subgraphs.append(subgraphs)
            # print(subgraphs.shape)
            batch_n.append(torch.full([len(neighbors)], i, dtype=torch.long,device=device))
            batch_g.append(torch.full([subgraphs.shape[1]], i, dtype=torch.long,device=device))
        
        all_neighbors = torch.cat(all_neighbors)
        all_subgraphs = torch.cat(all_subgraphs,dim=1)
        batch_n = torch.cat(batch_n)
        batch_g = torch.cat(batch_g)

        return nodes,all_neighbors,batch_n,all_subgraphs,batch_g

def get_labeled(train_mask, edge_index, hop, device):

    train_idx = np.arange(len(train_mask))[train_mask.cpu().numpy()]
    return get_batch(train_idx,edge_index,hop,device)

class TrainLoader(object):

    def __init__(self, train_mask, edge_index, sample_size, N_node=50, N_edge=100,hop=1, device=None):

        self.edge_index = edge_index

        self.node_idx = np.arange(len(train_mask))[torch.logical_not(train_mask).cpu().numpy()]
        self.train_mask = train_mask
        self.sample_size=sample_size
        self.hop = hop
        self.N_edge = N_edge

        if device == None:
            device = train_mask.device
        self.device = device

    def get_unlabeled(self):

        hop = self.hop
        device = self.device
        edge_index = self.edge_index 
        idx = np.random.choice(self.node_idx,self.sample_size,replace=False)

        neg_idx = idx.copy()
        np.random.shuffle(neg_idx)

        nodes = torch.tensor(idx, dtype=torch.long, device=device)
        neg_nodes = torch.tensor(neg_idx, dtype=torch.long, device=device)

        all_subgraphs = {}

        for i,node in enumerate(neg_idx):
            neighbors, subgraphs,_,_=k_hop_subgraph(int(node),hop,edge_index)

            subgraphs = subgraphs[:,subgraphs[1]>=subgraphs[0]]

            all_subgraphs[node] = subgraphs

        pos_subgraphs = []
        neg_subgraphs = []

        p_n = 0
        n_n = 0

        all_x_pos = []
        all_y_pos = []

        all_x_neg = []
        all_y_neg = []

        for idx, neg_idx in zip(idx, neg_idx):

            pos_subgraphs.append(all_subgraphs[idx])
            neg_subgraphs.append(all_subgraphs[neg_idx])

            p_l = all_subgraphs[idx].shape[1]
            n_l = all_subgraphs[neg_idx].shape[1]

            x_pos,y_pos = np.meshgrid(np.arange(p_n, p_n+p_l), np.arange(p_n, p_n+p_l), indexing='ij')
            
            x_pos = torch.tensor(x_pos.flatten(), dtype=torch.long, device=device)
            y_pos = torch.tensor(y_pos.flatten(), dtype=torch.long, device=device)
            
            all_x_pos.append(x_pos)
            all_y_pos.append(y_pos)

            x_neg,y_neg = np.meshgrid(np.arange(p_n, p_n+p_l), np.arange(n_n, n_n+n_l), indexing='ij')

            x_neg = torch.tensor(x_neg.flatten(), dtype=torch.long, device=device)
            y_neg = torch.tensor(y_neg.flatten(), dtype=torch.long, device=device)

            all_x_neg.append(x_neg)
            all_y_neg.append(y_neg)

            p_n += p_l
            n_n += n_l


        pos_subgraphs = torch.cat(pos_subgraphs,dim=1)
        neg_subgraphs = torch.cat(neg_subgraphs,dim=1)
        x_pos = torch.cat(all_x_pos)
        y_pos = torch.cat(all_y_pos)
        x_neg = torch.cat(all_x_neg)
        y_neg = torch.cat(all_y_neg)

        return nodes, neg_nodes, pos_subgraphs, neg_subgraphs, x_pos, y_pos, x_neg, y_neg
        
    def get_pert_edge(self, p=0.1):

        from torch_geometric.utils import negative_sampling,to_undirected

        neg_edge = negative_sampling(self.edge_index, num_neg_samples=int(p*self.edge_index.shape[1]), force_undirected=True)
        neg_edge = neg_edge[:,(neg_edge[0]>=0)&(neg_edge[1]>=0)]
        neg_edge = neg_edge[:,(neg_edge[0]<len(self.train_mask))&(neg_edge[1]<len(self.train_mask))]
        unique_edge =  self.edge_index[:,self.edge_index[0]>=self.edge_index[1]]
        index = np.random.choice(unique_edge.shape[1], int((1-p)*unique_edge.shape[1]), replace=False)

        common_edge = unique_edge[:,index]

        pert_edge_index = to_undirected(torch.cat([common_edge, neg_edge], dim=1))

        return pert_edge_index, common_edge

    def get_train(self):
        
        node_idx = np.random.choice(len(self.train_mask),self.sample_size,replace=False)
        # unlabeled_nodes = get_batch(idx, self.edge_index, self.hop, self.device)

        nodes = torch.tensor(node_idx, dtype=torch.long, device=self.device)


        unique_edge =  self.edge_index[:,self.edge_index[0]>=self.edge_index[1]]

        pert_edge_index, common_edge = self.get_pert_edge()

        edge_index = np.random.choice(common_edge.shape[1], 4*self.sample_size, replace=False)
        edges = common_edge[:,edge_index]

        # neg_edges = self.edge_index
        neg_unique_edge = pert_edge_index[:, pert_edge_index[0]>=pert_edge_index[1]]
        
        neg_edges = []
        for index in edge_index:
            mask = (neg_unique_edge[0]!=common_edge[0, index]) | (neg_unique_edge[1] != common_edge[1,index])

            neg_edges_row = np.random.choice(neg_unique_edge.shape[1]-1, self.N_edge, replace=True)
            neg_edges.append(neg_unique_edge[:,mask][:,neg_edges_row])
        neg_edges = torch.stack(neg_edges,dim=-1)

        return nodes, edges, neg_edges, pert_edge_index

class TestLoader:

    def __init__(self, test_mask, edge_index, sample_size, hop=1, device=None):
        self.edge_index = edge_index
        self.sample_size = sample_size
        self.hop = hop
        self.test_idx = np.arange(len(test_mask))[test_mask.cpu().numpy()]
        if device == None:
            device = test_mask.device
        self.device = device

        self.data = []
        for item in range(len(self)):
            idx = self.test_idx[sample_size*item:sample_size*(item+1)]
            self.data.append(get_batch(idx,self.edge_index,self.hop,self.device))
    
    def __len__(self):
        if len(self.test_idx) % self.sample_size==0:
            return len(self.test_idx)//self.sample_size
        else:
            return len(self.test_idx)//self.sample_size + 1

    def __getitem__(self, item):

        return self.data[item]


# %%
