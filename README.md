
# SE-GNN
A PyTorch implementation of "Towards Self-Explainable Graph Neural Network" (CIKM 2021). [[paper]](https://arxiv.org/pdf/2108.12055.pdf)
## Abstract
Graph Neural Networks (GNNs), which generalize the deep neural networks to graph-structured data, have achieved great success in modeling graphs. However, as an extension of deep learning for graphs, GNNs lack explainability, which largely limits their adoption in scenarios that demand the transparency of models. Though many efforts are taken to improve the explainability of deep learning, they mainly focus on i.i.d data, which cannot be directly applied to explain the predictions of GNNs because GNNs utilize both node features and graph topology to make predictions. There are only very few work on the explainability of GNNs and they focus on post-hoc explanations. Since post-hoc explanations are not directly obtained from the GNNs, they can be biased and misrepresent the true explanations. Therefore, in this paper, we study a novel problem of self-explainable GNNs which can simultaneously give predictions and explanations. We propose a new framework which can find $K$-nearest labeled nodes for each unlabeled node to give explainable node classification, where nearest labeled nodes are found by interpretable similarity module in terms of both node similarity and local structure similarity.  Extensive experiments on real-world and synthetic datasets demonstrate the effectiveness of the proposed framework for explainable node classification.
## Dataset
The **Cora, citeseer, and Pubmed** datasets will be automatically downloaded to `./data`. 

For **BA-Shape**, we are using the one that generated by [PGExplainer](https://github.com/flyingdoog/PGExplainer/tree/master/dataset)

For **Syn-Cora**, we release the synthetic dataset Syn-Cora in `./data/Syn-Cora.gpickle`. We can easily load the dataset by:
```
import networkx as nx
from torch_geometric.utils import from_networkx
G = nx.read_gpickle("./data/Syn-Cora.gpickle")
data = from_networkx(G).to(device)
``` 
In the Syn-Cora, each node/edge is given a node/edge role for ground truth of k-nearest labeled nodes and edge maching. Nodes/edges with the same role number should be matched.

The statistics of Syn-Cora are:

|Dataset|#Nodes| #Edges | #Features | #Classes|
|---|---|---|---|---|
|Syn-Cora|1895|2769|1433|7|

Note that the statistics are slightly different from the reported ones in the paper, because of using different random seeds. The results of this given syn-cora can be checked in the following section. The code to generate syn-cora is in `SynData.py`.



## Cite
If you find this repo to be useful, please cite our paper. Thank you.
```
@article{dai2021towards,
  title={Towards Self-Explainable Graph Neural Network},
  author={Dai, Enyan and Wang, Suhang},
  journal={arXiv preprint arXiv:2108.12055},
  year={2021}
}
```
# Reproduce the results

The results are not deterministic even with the all the seeds fixed due to the cacluation on GPU with PyG. 
So the obtained results can be different from the reported. Here, we provide the script and reference results to help reproduce the results of each section

## Requirements

```
networkx
torch==1.7.1
torch-geometric==1.7.2 
```

## On real-world datasets
After installation, you can clone this repository and run:
```
bash train_real.sh
```
Here are some results as reference
|Dataset| | | | |
|---|---|---|---|---|
| Pubmed | 0.789 | 0.795 | 0.805 | 0.802 |
| Citeseer | 0.7293 | 0.7494 | 0.7399 | 0.7322 |
| Cora | 0.8040 | 0.8120 | 0.7950 | 0.8010|
## On Syn-Cora
You can call this code to train model on Syn-Cora
```
bash train_syn.sh
```
Here are some results as reference
|Metric| | | | |
|---|---|---|---|---|
| Accuracy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000|
| Node acc | 0.9823| 0.9696|0.9783 | 0.9957|0.9913|
| Edge acc | 0.8096| 0.8104|0.7993 | 0.8543|0.8521|

## On BA-Shape
You can run this code to train the model on BA-Shape dataset:
```
bash train_BAshape.sh
```
Here are some results for reference:
|Metric| | | | |
|---|---|---|---|---|
| ROC AUC | 0.9728 | 0.9728 | 0.9784 | 0.9746 | 0.9759|
## Provided model for evaluation
We also provided trained models in `./checkpoint` for real-world datasets for evaluation to reproduce precision@k (Figure 3). You can call:
```
bash test_real.sh
```


