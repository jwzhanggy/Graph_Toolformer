
# Source code of the Graph-Toolformer Package for various graph reasoning Tasks.

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

ABout 15 different graph datasets are used and studied in graph-toolformer in the current version, which include

- Graph Property Reasoning Dataset: GPR
- Bibliographic Network Datasets: Cora, Pubmed, Citeseer
- Molecular Graph Datasets: Proteins, Mutag, NCI1, PTC
- Online Social Network Datasets: Twitter, Foursquare
- Recommender System Datasets: Amazon, Last-FM, Movielens
- Knowledge Graph Datasets: WorldNet, Freebase

### Dataset Formats.

The datasets are stored in binary format, which can be loaded with pickle, e.g., as follows:

```
import pickle
f = open('./cora', 'rb')
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
            'is_weighted': boolean
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
            'is_weighted': boolean
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
(Download URL: To be provided)

The koala folder covers the pre-trained graph models and pre-trained language models used in toolformer. Both the model code and pre-trained results/parameters are provided in the downloaded folder.

### Pre-trained Graph Models

- Graph Property Reasoning Model: toolx
- Bibliographic Network Reasoning Model: Graph-Bert
- Molecular Graph Reasoning Model: Model, SEG-Bert
- Online Social Network Reasoning Model: KMeans
- Recommender System Reasoning Model: BPR (Bayesian Personalized Ranking)
- Knowledge Graph Reasoning Model: TransE

### Pre-trained Language Models

- GPT-J 6B 8Bit
