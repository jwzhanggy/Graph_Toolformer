
# Graph-Toolformer Package for Graph Reasoning Demos

************************************************************************************************

## How to run the demo code?

After downloading this directory, also downloading the [graph_datasets.zip](https://drive.google.com/file/d/1lC23j9RYMb44JRJybxIpUtxuQ2lW58n_/view?usp=sharing), [koala/graph_models.zip](https://drive.google.com/file/d/1Wpw2DBa2p_PG_od3AVGZd5GvoKhCM2mH/view?usp=share_link) and [koala/language_models.zip](https://drive.google.com/file/d/1wja3_aBbWg6YztB7gNna_2FuN10dWjiZ/view?usp=sharing), installing the [conda environment](https://github.com/jwzhanggy/Graph_Toolformer/blob/main/environment.yml), you can just go ahead to run the code with the following command to start the demo

```
python3 ./src/Framework_Graph_Toolformer.py
```
You can type in inputs which are similar to the prompt inputs, and the model will carry out the reasoning task and return the otuputs. The reasoning process will call both the LLMs and GNN models, so generating the output will take some time. The GNN models are pre-trained according to the previous papers, and the LLMs is fine-tuned with the code in the LLM_Tunign directory based on the prompt datasets.

By changing the "if 0" to "if 1" in the bottom main function of Framework_Graph_Toolformer.py, you can try different reasoning tasks.

We will also allow users to use command lines to interact with the Graph-toolformer framework. The CLI will be added shortly.

************************************************************************************************

## Package Contents
The package code folder contains three sub_folders organized as follows:

```
./Graph_Toolformer_Package
./Graph_Toolformer_Package/graph_datasets
./Graph_Toolformer_Package/koala
./Graph_Toolformer_Package/src
```

The src folder is uploaded to github, and the remaining two folders are too big and can be downloaded with the URL provided as follows. Instructions about how to use the package will be provided shortly after the cli is ready.

## Graph Datasets used in Graph-Toolformer
(Download URL: https://drive.google.com/file/d/1lC23j9RYMb44JRJybxIpUtxuQ2lW58n_/view?usp=sharing)

About 15 different graph datasets are used and studied in graph-toolformer in the current version, which include

- **Graph Property Reasoning Dataset**: GPR
- **Bibliographic Network Datasets**: Cora, Pubmed, Citeseer
- **Molecular Graph Datasets**: Proteins, Mutag, NCI1, PTC
- **Online Social Network Datasets**: Twitter, Foursquare
- **Recommender System Datasets**: Amazon, Last-FM, Movielens
- **Knowledge Graph Datasets**: WordNet, Freebase

### Dataset Formats.

The datasets are stored in binary format, which can be loaded with pickle, e.g., the GPR dataset can be loaded as follows:

```
import pickle
f = open('./gpr', 'rb')
dataset = pickle.load(f)
f.close()
print(dataset.keys())
```

#### For the datasets with 1 single large-scale graph/network (including, Cora, Pubmed, Citeseer; Twitter, Foursquare; Amazon, Last-FM, Movielens; WordNet, Freebase), the loaded "dataset" is organized with a python dictionary with the following format:

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

#### For the datasets with multiple graph instances (GPR; Proteins, Mutag, NCI1, PTC), the loaded "dataset" is organized with a python dictionary with the following format:

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
(Download URL: https://drive.google.com/file/d/1Wpw2DBa2p_PG_od3AVGZd5GvoKhCM2mH/view?usp=share_link)

The koala folder covers the pre-trained graph models and pre-trained language models used in toolformer. Both the model code and pre-trained results/parameters are provided in the downloaded folder.

### Pre-trained Graph Models

6 different pre-trained graph models are included in the Graph-Toolformer framework for different graph reasoning tasks, which are listed as follows:

- **Graph Property Reasoning Model**: toolx
- **Bibliographic Network Reasoning Model**: Graph-Bert
- **Molecular Graph Reasoning Model**: SEG-Bert
- **Online Social Network Reasoning Model**: KMeans
- **Recommender System Reasoning Model**: BPR (Bayesian Personalized Ranking)
- **Knowledge Graph Reasoning Model**: TransE

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
```
- community(graph): return the community label for all user nodes in the social network
- community(graph, node): return the community label for specify input user nodes
- community_count(graph): return the number of detected communities
- community_avg_size(graph): return the average size of detected communities
- community_max_size(graph): return the max size of detected communities
- community_size(graph, node): return the size of community that the input user node belongs to
- common_community_check(graph, node1, node2): check if user1 and user2 belong to the same community or not
```
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
- LLaMA (To be Done)

#### GPT-J 6B 8bit

The EleutherAI's GPT-J (6B) is a transformer model trained using Ben Wang's ``[Mesh Transformer JAX](https://www.eleuther.ai/artifacts/mtj)'' that has 6 billion parameters. To load GPT-J in float 32, it will require 22+GB RAM to load the model and the fine-tuning will require at least 4x RAM size. To further lower-down the RAM consumption for GPT-J (6B), researchers also propose to quantize it with 8-bit weights, which allows scalable fine-tuning with LoRA (Low-Rank Adaptation) and 8-bit Adam and GPU quantization from bitsandbytes. The Graph-Toolformer (GPT-J 6B, 8bit) model will use GPT-J 6B 8bit as the base model for fine-tuning.

