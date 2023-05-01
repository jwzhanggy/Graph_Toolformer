# Graph-Toolformer
(project in-preparation)

![framework](./figures/framework.png)

### Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT

Paper URL at IFMLab: http://www.ifmlab.org/files/paper/graph_toolformer.pdf

Paper URL at arxiv: https://arxiv.org/pdf/2304.11116.pdf

Paper description in Chinese: [文章中文介绍](./中文介绍)

### Project Conda Environment

See the shared [environment.yml](./environment.yml) file. Create a local environment at your computer with command 
```
conda env create -f environment.yml
```
For the packages cannot be instead with the above conda command, you may consider to install via pip instead.

### References 
(also add \usepackage{hyperref} into your paper for adding the url address in the reference)

```
@article{zhang2023graph,
  title={Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT},
  author={Zhang, Jiawei},
  note = {\url{http://www.ifmlab.org/files/paper/graph_toolformer.pdf}},
  year={2023},
  note = "[Online; Published 10-April-2023]"
}
```

************************************************************************************************

## Tasks to be done

- [x] Polish the framework: 8/8 done
  - [x] add working memory module
  - [x] add query parser module
  - [x] add query excutor module
  - [x] add graph dataset hub
  - [x] add graph model hub
  - [x] add graph reasoning task hub
  - [x] add llm model hub
  - [x] add llm trainer module
- [ ] Expand the framework: 3/4 done
  - [x] Include graph datasets
    - [x] graph property dataset
    - [x] bibliographic networks: cora, pubmed, citeseer
    - [x] molecular graphs: proteins, nci1, mutag, ptc
    - [x] social networks: twitter, foursquare
    - [x] recommender system: amazon, last.fm, movielens  
    - [x] knowledge graphs: wordnet, freebase 
  - [x] Add pre-trained graph models
    - [x] Toolx
    - [x] Graph-Bert
    - [x] SEG-Bert
    - [x] KMeans Clustering
    - [x] BPR
    - [x] TransE
  - [x] Include graph reasoning tasks
    - [x] graph property reasoning
    - [x] bibliographic paper topic reasoning
    - [x] molecular graph function reasoning
    - [x] social network community reasoning
    - [x] recommender system reasoning
    - [x] knowledge graph reasoning
  - [ ] Add and test more LLMs
    - [x] GPT-J
    - [ ] LLaMA
    - [ ] GPT-2
    - [ ] OPT
    - [ ] Bloom
- [ ] Release the framework and service: 0/5 done
  - [ ] Polish and release the datasets
  - [ ] Implement the CLI with GUI panel
  - [ ] Provide the demo for graph reasoning
  - [ ] Add API for customerized graph reasoning
  - [ ] Release web access and service


