
# Graph-Toolformer Package for Graph Reasoning

## Package Contents
The package code folder contains three sub_folders organized as follows:

```
./Graph_Toolformer_Package
./Graph_Toolformer_Package/graph_datasets
./Graph_Toolformer_Package/koala
./Graph_Toolformer_Package/src
```

The src folder is uploaded to github, and the remaining two folders are too big and can be downloaded with the URL provided as follows.

## Graph Datasets used in Graph-Toolformer
(Download URL: https://drive.google.com/file/d/1lC23j9RYMb44JRJybxIpUtxuQ2lW58n_/view?usp=sharing)

About 15 different graph datasets are used and studied in graph-toolformer in the current version, which include

- Graph Property Reasoning Dataset: GPR
- Bibliographic Network Datasets: Cora, Pubmed, Citeseer
- Molecular Graph Datasets: Proteins, Mutag, NCI1, PTC
- Online Social Network Datasets: Twitter, Foursquare
- Recommender System Datasets: Amazon, Last-FM, Movielens
- Knowledge Graph Datasets: WorldNet, Freebase

### Dataset Formats.

The datasets are stored in binary format, which can be loaded with pickle, e.g., the GPR dataset can be loaded as follows:

```
import pickle
f = open('./gpr', 'rb')
dataset = pickle.load(f)
f.close()
print(dataset.keys())
```

#### For the datasets with 1 single large-scale graph/network (including, Cora, Pubmed, Citeseer; Twitter, Foursquare; Amazon, Last-FM, Movielens; WorldNet, Freebase), the loaded "dataset" is organized with a python dictionary the following format:

```
dataset = {
        "data_profile": {
            'name': dataset_name,
            'order': node_number,
            'size': link_number,
            'is_directed': boolean,
            'is_weighted': boolean,
            # besides the above profile information, for some data, we will also include some other attributes, like feature vector dimensions, label space dimension, etc., in the data profile dict.
        },
        "nodes": {
          node_id: {'features': feature, 'label': label,}
        },
        "links": {
          node_pair: {'features': feature, 'label': label,}
        }
    }
```

#### For the datasets with multiple graph instances (GPR; Proteins, Mutag, NCI1, PTC), the loaded "dataset" is organized with a python dictionary the following format:

```
dataset = {
        "data_profile": {
            'name': dataset_name,
            'graph_number': graph_number,
            'is_directed': boolean,
            'is_weighted': boolean,
            # besides the above profile information, for some data, we will also include some other attributes, like feature vector dimensions, label space dimension, etc., in the data profile dict.
        },
        'graph_set': {
            graph_id: {
                'nodes': node_set,
                'links': link_set,
                'label': graph_instance_label,
            }
        }
    }
```

## Pre-Trained Graph Models and fine-tuned LLMs used in Graph-Toolformer
(Download URL: https://drive.google.com/file/d/15dMT96MHES56hV1MNlCrf2RWKPnyecao/view?usp=sharing)

The koala folder covers the pre-trained graph models and pre-trained language models used in toolformer. Both the model code and pre-trained results/parameters are provided in the downloaded folder.

### Pre-trained Graph Models

- Graph Property Reasoning Model: toolx
- Bibliographic Network Reasoning Model: Graph-Bert
- Molecular Graph Reasoning Model: Model, SEG-Bert
- Online Social Network Reasoning Model: KMeans
- Recommender System Reasoning Model: BPR (Bayesian Personalized Ranking)
- Knowledge Graph Reasoning Model: TransE

#### Toolx for Graph Property Reasoning

The current toolx model is implemented based on networkx, and toolx will implement different functions to calculate different graph properties mentioned in the paper, which are also listed as follows:
```
- order
- size
- density
- eccentricity
- radius
- diameter
- center
- shortest_path
- avg_path_length
- min_path_length
- max_path_length
- periphery
```

#### Graph-Bert for Bibliographic Network Paper Topic Reasoning

The Graph-Bert model was proposed in paper entitled "[Graph-Bert: Only Attention is Needed for Learning Graph Representations](https://arxiv.org/abs/2001.05140)". 

The Graph-Bert will be used to implement the bibliographic network paper topic inference function, which will be reduced to the node classification task:
```
- node_classification in GraphBertNodeClassification.py
```
The model has been pre-trained on the cora, pubmed, citeseer datasets already based on the identical train/test sets introduced in the paper. Both the model code and the pre-trained model parameter checkpoints are provided.

As to the original source code, readers may consider to refer to the repository (https://github.com/jwzhanggy/Graph-Bert) for more information.

#### SEG-Bert for Molecualr Graph Function Reasoning

The SEG-Bert model was proposed in the paper entitled "[Segmented Graph-Bert for Graph Instance Modeling
](https://arxiv.org/abs/2002.03283)".

The SEG-Bert will be used to implement the molecular graph function inference function, which will be reduced to the graph classification task:
```
- graph_classification in SegmentedGraphBertGraphClassification.py
```
The model has been pre-trained on the proteins, mutag, nci1 and ptc datasets already based on the identical train/test sets introduced in the paper. Both the model code and the pre-trained model parameter checkpoints are provided.

As to the original source code, readers may consider to refer to the repository ([https://github.com/jwzhanggy/Graph-Bert](https://github.com/jwzhanggy/SEG-BERT)) for more information.

#### KMeans for Social Network Community Reasoning

To detect the social network community, based on the social network structure (adjacency matrix), we calculate the nodes' pairwise common neighbor numbers to define their closeness, which will be fed to KMeans for community detection.

#### BPR

The BPR model was proposed in the paper entitled "[BPR: Bayesian personalized ranking from implicit feedback](https://arxiv.org/pdf/1205.2618.pdf)".

The BPR model will be used to implement the social network community detection functions, which will be reduced to the graph partition/clustering tasks:
```
- recommendation in BPR.py
- topk_recommendation in BPR.py
```

#### TransE

The TransE model was proposed in the paper entitled "[Transition-based Knowledge Graph Embedding with Relational Mapping
Properties](https://aclanthology.org/Y14-1039.pdf)".

The TransE will be used to implement the knowledge graph entity/relation searching functions, which will be reduced to the graph searching tasks:
```
- search_head_entity in TransE.py
- search_tail_entity in TransE.py
- search_relation in TransE.py
```

### Pre-trained Language Models

- GPT-J 6B 8bit 
- LLaMA (To be added)
- Bloom (To be added)
- OPT (To be added)
- GPT2 (To be added)

#### GPT-J 6B 8bit


