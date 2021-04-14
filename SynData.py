#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import k_hop_subgraph,is_undirected, subgraph
from torch_geometric.datasets import Planetoid


# %%
from torch_geometric.utils import remove_isolated_nodes
class SynData:

    def __init__(self, data):
        self.data = data
        self.dist = []
        for i in range(data.y.max()+1):
            self.dist.append(data.x[data.y==i].sum(dim=0))

    def plot_subgraph(self, subgraphs, y):
        G = nx.Graph()
        G.add_edges_from(subgraphs)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_color=y, node_size=800, cmap=plt.cm.tab10)
        plt.show()

    def add_noise(self, idx, noise_rate=0.1):

        x = self.data.x[idx].numpy().copy()

        p = np.ones([len(x)])

        p[x==1]=0
        p = p/p.sum()
        size = int(x.sum()*noise_rate)
        noise = np.random.choice(len(p), size=size, p=p, replace=False)
        x[noise] = 1
        noise = np.random.choice(x.nonzero()[0], size=size, replace=False)
        x[noise] = 0

        return x

    def syn_local(self, start, neighbors, subgraphs, node_start, edge_start, number=10, noise_rate=0.4):

        
        data = self.data
        
        y = data.y[neighbors].numpy()
        subgraphs = subgraphs[:,subgraphs[0]<subgraphs[1]].T.numpy()

        #self.plot_subgraph(subgraphs, y)

        node_roles = np.arange(node_start, node_start+len(neighbors))
        edge_roles = np.arange(edge_start, edge_start+len(subgraphs))

        neighbor_x = data.x[neighbors].numpy()
        graphs = []
        for i in range(number):
            G = nx.Graph()
            for j, idx in enumerate(neighbors):
                new_node_idx = start + i*len(neighbors) + j
                if i == 0:
                    x = data.x[idx].numpy()
                else:
                    x = self.add_noise(idx, noise_rate)
                #print(np.abs(x-data.x[idx].numpy()).sum())
                G.add_node(new_node_idx, x=x, y=int(data.y[idx]), node_role=node_roles[j])
            for edge, role in zip(subgraphs, edge_roles):
                G.add_edge(edge[0] + i*len(neighbors) + start, edge[1] + i*len(neighbors) + start, edge_role=role)
            graphs.append(G)

        return graphs, start + number*len(neighbors), node_start + len(neighbors), edge_start + len(subgraphs)

    def syn_random_basis(self, n, d, start=0, node_start=0, edge_start=0):
        data = self.data
        m = d*n
        label = int(data.y.max()) + 1
        p = data.x.mean(dim=0).numpy()
        p[:10] = 0.7
        G = nx.gnm_random_graph(n, m, seed=15, directed=False)
        nids = sorted(G)
        mapping = {nid: start + i for i, nid in enumerate(nids)}
        G = nx.relabel_nodes(G, mapping)
        for i, node in enumerate(G.nodes):
            G.nodes[node]['x']= np.random.binomial(1,p)
            G.nodes[node]['node_role'] = node_start
            G.nodes[node]['y'] = label

        for edge in G.edges:
            G.edges[edge]['edge_role']= edge_start
        
        return G
    
    def syn_real_basis(self, n, remove_nodes, start=0, node_start=0, edge_start=0):
        data = self.data

        remain_nodes = list(set(np.arange(len(data.x))) - set(remove_nodes))
        remain_nodes = list(sorted(np.random.choice(remain_nodes, n, False)))

        G = nx.Graph()

        edge_index,_ = subgraph(remain_nodes, data.edge_index, relabel_nodes=True)
        _, _, node_mask = remove_isolated_nodes(edge_index, num_nodes=len(remain_nodes))
        
        remain_nodes = list(np.asarray(remain_nodes)[node_mask])

        edge_index,_ = subgraph(remain_nodes, data.edge_index, relabel_nodes=True)

        for i, idx in enumerate(remain_nodes):
            G.add_node(i+start, x=data.x[idx].numpy(), node_role=node_start, y=int(data.y[idx]))
        G.add_edges_from(edge_index.T.numpy() + start, edge_role=edge_start)

        return G
    

    def syn_graph(self, n_basis=300, basis_type='random',nb_shapes=1, nb_copy=10, hop=2, noise_rate=0.0, connect=1, seed=15):
        
        np.random.seed(seed)

        nb_class = int(self.data.y.max() + 1)
        node_start = 1
        edge_start = 1
        graphs = []
        remove_nodes = []
        for label in range(nb_class):
            n = 0
            indices = (self.data.y == label).nonzero().flatten().numpy()
            np.random.shuffle(indices)
        
            for idx in indices:
                neighbors, subgraphs,_,_=k_hop_subgraph(int(idx),hop,self.data.edge_index,relabel_nodes=True)
                
                
                overlap = set(list(neighbors.numpy())) & set(remove_nodes)
                
                if len(neighbors)>=5 and len(neighbors)<=20 and len(overlap)==0:
                    #print(self.data.y[idx])
                    #self.plot_subgraph(subgraphs.T.numpy(), self.data.y[neighbors].numpy())
                    G, start, node_start, edge_start = self.syn_local(0, neighbors, subgraphs,\
                                                                node_start, edge_start, nb_copy, noise_rate)
                                                            
                    n += 1
                    graphs += G
                    remove_nodes += list(neighbors.numpy())
                if n == nb_shapes:
                    break
        start = 0

        if basis_type=='random':
            basis = self.syn_random_basis(n_basis, 2, start)
        else:
            basis = self.syn_real_basis(n_basis * 2, remove_nodes, start)

        start += basis.number_of_nodes()
        for motif in graphs:
            mapping = {nid: start + i for i, nid in enumerate(motif.nodes)}
            motif = nx.relabel_nodes(motif, mapping)
            start += motif.number_of_nodes()

            basis.add_nodes_from(motif.nodes(data=True))
            basis.add_edges_from(motif.edges(data=True))
            u = np.random.choice(list(basis), connect, False)
            v = np.random.choice(list(motif), connect, False)
            for i,j in zip(u,v):
                basis.add_edge(i,j, edge_role=0)

        return basis
    
    def get_train_test_split(self, G):

        from torch_geometric.utils import from_networkx
        from sklearn.model_selection import train_test_split

        syn_data = from_networkx(G)
        np.random.seed(15)
        motif_idx = (syn_data.node_role > 0).nonzero().flatten().numpy()
        motif_node_role = syn_data.node_role[motif_idx].numpy()
        train_motif_idx, test_idx = train_test_split(motif_idx, train_size=0.3, test_size=0.5, stratify=motif_node_role)
        val_idx = list(set(list(motif_idx))-set(list(train_motif_idx))-set(list(test_idx)))

        basis_idx = (syn_data.node_role == 0).nonzero().flatten().numpy()
        train_basis_idx = np.random.choice(basis_idx, size=int(0.2*len(basis_idx)), replace=False)

        train_mask = torch.zeros([syn_data.num_nodes],dtype=torch.bool)
        train_mask[train_motif_idx]=True
        train_mask[train_basis_idx]=True

        val_mask = torch.zeros([syn_data.num_nodes],dtype=torch.bool)
        val_mask[val_idx]=True

        test_mask = torch.zeros([syn_data.num_nodes],dtype=torch.bool)
        test_mask[test_idx] = True

        syn_data.train_mask = train_mask
        syn_data.val_mask = val_mask
        syn_data.test_mask = test_mask

        return syn_data
#%%
'''
dataset = Planetoid(root='./data/',name='Cora')
data = dataset.data
generator = SynData(data)
G = generator.syn_graph(n_basis=100,basis_type='random')
data = generator.get_train_test_split(G)
#%%
features = data.x
idx = -1
score = features[idx] @ features.T
_, indices = score.topk(3)
data.node_role[indices]
#%%
data.node_role[idx]
# %%
'''