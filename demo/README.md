## Source code of the Graph-Toolformer Framework Demo for various graph reasoning Tasks.

## Graph Datasets used in Graph-Toolformer
(Download URL: https://drive.google.com/file/d/1lC23j9RYMb44JRJybxIpUtxuQ2lW58n_/view?usp=sharing)

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

## Pre-Trained Graph Models used in Graph-Toolformer
(Download URL: )

