#%%
import torch
import numpy as np
from torch_geometric.utils import k_hop_subgraph,is_undirected

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

    def __init__(self, train_mask, edge_index, sample_size, hop=1, device=None):

        self.edge_index = edge_index
        self.node_idx = np.arange(len(train_mask))[torch.logical_not(train_mask).cpu().numpy()]
        self.sample_size=sample_size
        self.hop = hop

        if device == None:
            device = train_mask.device
        self.device = device
    
    def __iter__(self):
        return self
    
    def __next__(self):
        idx = np.random.choice(self.node_idx,self.sample_size,replace=False)
        return get_batch(idx,self.edge_index,self.hop,self.device)

class TestLoader:

    def __init__(self, test_mask, edge_index, sample_size, hop=1, device=None):
        self.edge_index = edge_index
        self.sample_size = sample_size
        self.hop = hop
        self.test_idx = np.arange(len(test_mask))[test_mask.cpu().numpy()]
        if device == None:
            device = test_mask.device
        self.device = device
    
    def __len__(self):
        return len(self.test_idx)//self.sample_size

    def __getitem__(self, item):
        sample_size = self.sample_size
        idx = self.test_idx[sample_size*item:sample_size*(item+1)]

        return get_batch(idx,self.edge_index,self.hop,self.device)
# # %%
# loader = TestLoader(data.test_mask,data.edge_index,100)
# train_data = TrainLoader(data.train_mask, data.edge_index, 10)
# label_nodes = get_labeled(data.train_mask, data.edge_index, 1, device=data.train_mask.device)
# # %%
# np.random.seed(10)
# train_dataloader = iter(train_data)

# # %%
# next(train_dataloader)

# %%
# t = get_batch([0],data.edge_index,1,device)
from utils import build_graph
def gen_syn1(nb_shapes=80, width_basis=200, m=1):
    """ Synthetic Graph #1:
    Start with Barabasi-Albert graph and attach house-shaped subgraphs.
    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here 'Barabasi-Albert' random graph).
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  number of edges to attach to existing node (for BA graph)
    Returns:
        G                 :  A networkx graph
        role_id           :  A list with length equal to number of nodes in the entire graph (basis
                          :  + shapes). role_id[i] is the ID of the role of node i. It is the label.
        name              :  A graph identifier
    """
    basis_type = "ba"
    list_shapes = [["house"]] * int(nb_shapes/2)  + [["star", 4]] * int(nb_shapes/2)

    # plt.figure(figsize=(8, 6), dpi=300)

    G, role_id, _ = build_graph(
        width_basis, basis_type, list_shapes, start=0, m=m
    )
    # G = perturb([G], 0.01)[0]

    # if feature_generator is None:
    #     feature_generator = featgen.ConstFeatureGen(1)
    # feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)
    return G, role_id, name

# %%

from torch_geometric.utils import from_networkx
from sklearn.model_selection import train_test_split
# def get_syn(nb)

def get_syn():

    np.random.seed(15)
    G, labels,_ = gen_syn1()

    indices = np.arange(len(labels))
    data = from_networkx(G)
    train_masks=[]
    val_masks=[]
    test_masks = []

    for seed in range(15,20):
        train_indices, test_indices, train_labels,test_labels = train_test_split(indices,labels,test_size=0.5,stratify=labels,random_state=seed)
        train_indices, val_indices, train_labels,val_labels = train_test_split(train_indices,train_labels,test_size=0.5,stratify=train_labels,random_state=seed)

        train_mask = torch.zeros([len(labels)],dtype=torch.bool)
        train_mask[train_indices]=True

        val_mask = torch.zeros([len(labels)],dtype=torch.bool)
        val_mask[val_indices]=True

        test_mask = torch.zeros([len(labels)],dtype=torch.bool)
        test_mask[test_indices] = True

        train_masks.append(train_mask.unsqueeze(dim=1))
        val_masks.append(val_mask.unsqueeze(dim=1))
        test_masks.append(test_mask.unsqueeze(dim=1))

    data.x = torch.eye(len(labels))
    data.y = torch.LongTensor(labels)
    data.train_mask = torch.cat(train_masks,dim=1)
    data.val_mask = torch.cat(val_masks,dim=1)
    data.test_mask = torch.cat(test_masks,dim=1)
    return data

# %%
