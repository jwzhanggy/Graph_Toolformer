# Graph-Toolformer

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

************************************************************************************************

## Organization of the code?

By adding the needed dataset, the model can run by calling "python gtoolformer_gptj_script.py".

A simpler template of the code is also available at http://www.ifmlab.org/files/template/IFM_Lab_Program_Template_Python3.zip

### The whole program is divided into five main parts:

(1) data.py (for data loading and basic data organization operators, defines abstract method load() )

(2) method.py (for complex operations on the data, defines abstract method run() )

(3) result.py (for saving/loading results from files, defines abstract method load() and save() )

(4) evaluate.py (for result evaluation, defines abstract method evaluate() )

(5) setting.py (for experiment settings, defines abstract method load_run_save_evaluate() )

The base class of these five parts are defined in ./code/base_class/, they are all abstract class defining the templates and architecture of the code.

The inherited class are provided in ./code, which inherit from the base classes, implement the abstract methonds.

## Detailed information on funtional classes?

### a. data

(1) DatasetLoader.py (for dataset loading)


### b. method

(1) Method_Graph_Toolformer_GPTJ.py (Graph-Toolformer Implementation with GPT-J 6b 8bit)

(2) Module_GPTJ_8bit.py (The GPT-J module quantization with 8 bit)


### c. result

(1) ResultSaving.py (for saving results to file)

(2) Result_Loader.py (for loading result from file)


### d. evaluate

(1) Evaluate_Metrics.py (implementation of evaluation metrics used in this experiment, include rouge scores, bleu scores, etc.)

### e. setting

(1) Setting_Regular.py (defines the interactions and data exchange among the above classes)

************************************************************************************************
