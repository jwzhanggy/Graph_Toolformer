# Graph-Toolformer

### 🟢 May 10: Updates
- **Update 1**: The paper has been updated, and the new version of the paper can be found [via this link](http://www.ifmlab.org/files/paper/graph_toolformer.pdf). The paper at arxiv will also be updated shortly.
- **Update 2**: The prompt dataset has been updated to correct some typos. The LLM fine-tuned checkpoint has been updated to reflect the changes to the prompts. The source code of the graph_toolformer_package has been updated for multiple continuous reasoning with the demo.
- **Update 3**: We add more descriptions about code organization and usage instructions:
    - [Organization of the project source code, data and model checkpoints](https://github.com/jwzhanggy/Graph_Toolformer/tree/main#organization-of-the-project-source-code-data-and-model-checkpoints)
    - [How to play with the code?](https://github.com/jwzhanggy/Graph_Toolformer/tree/main#how-to-play-with-the-code)

### 🟢 May 4: Updates
- **Update 1**: All the datasets and model checkpoints have been publicized, some large-sized data are shared via google drive
    - Datasets and Model Checkpoints (about 5GB): [See this page](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/data)
- **Update 2**: Source code of both Graph-toolformer LLM Tuning and Graph Reasoning Demo are released
    - LLMs Fine-Tuning code with Prompts: [See this page](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/LLM_Tuning)
    - Graph-toolformer Demo code: [See this page](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/Graph_Toolformer_Package)

************************************************************************************************

![framework](./figures/framework.png)

### Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT

Paper URL at IFMLab: http://www.ifmlab.org/files/paper/graph_toolformer.pdf

Paper URL at arxiv: https://arxiv.org/pdf/2304.11116.pdf

Paper description in Chinese: [文章中文介绍](./中文介绍)

### References 
(also add \usepackage{hyperref} into your paper for adding the url address in the reference)

```
@article{zhang2023graph,
    title={Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT},
    author={Zhang, Jiawei},
    year={2023},
    eprint={2304.11116},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
}
```

************************************************************************************************
## Organization of the project source code, data and model checkpoints

### Source code

This project source code is divided into two directories:
- **[LLM_Tuning](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/LLM_Tuning)**: Language model fine-tuning code with graph reasoning prompt dataset
- **[Graph_Toolformer_Package](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/Graph_Toolformer_Package)**: The Graph-Toolformer reasoning demo code, which will load the fine-tuned LLMs (tuned by the LLM_Tuning code) and the other pre-trained GNN models.

### Dataset

The datasets used in this paper inclode both the the generated graph reasoning prompt datasets, and the raw graph benchmark datasets
- **[Prompt Datasets](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/LLM_Tuning/prompt)**: The graph reasoning prompts created in this paper for LLM fine-tuning. The prompt dataset has been included in the LLM_Tuning directory already.
- **[Graph Raw Datasets](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/Graph_Toolformer_Package/graph_datasets)**: The 15 graph benchmark datasets used in this paper. The graph raw dataset (about 100MB) should be download from the google drive.

### Model checkpoints

The pre-trained/fine-tuned model checkpoints released by this project also have two parts
- **[Fine-tuned LLM Checkpoint](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/Graph_Toolformer_Package/koala/language_models)**: The checkpoint of fine-tuned language model by the above LLM_Tuning code, readers can either tune the LLMs by yourselves or use the provided checkpoint by us. If you plan to use the checkpoint released by us, you may need to download it from the Google drive, and the zip file is about 5GB.
- **[Pre-trained GNN Checkpoints](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/Graph_Toolformer_Package/koala/graph_models)**: We use 5 different graph models in this project, the checkpoints of the pre-trained GNNs are also provided. The readers can either pre-train their own graph models, or use the pre-trained graph model checkpoints released by us, which can be downloaded from Google drive. The zip file is about 100MB, and the upziped folder can be about 5GB.

************************************************************************************************

## How to play with the code?

### Environment setup

First of all, as mentioned above, please set up the environment for running the code. We recommend you create the code environment with the [environment.yml file](https://github.com/jwzhanggy/Graph_Toolformer/blob/main/environment.yml) shared from us. You can create the environment with the following command

```
conda env create -f environment.yml
```

For the packages cannot be installed with the above conda command, you may consider to install via pip.

### Play with LLM_Tuning code

After downloading the LLM_Tuning directory, installing the conda environment ([**see this file**](https://github.com/jwzhanggy/Graph_Toolformer/blob/main/environment.yml)), you can just go ahead to run the code with the following command to start the LLMs fine-tuning with the prompt datasets:

> **Note**
> 
> **1.** To avoid getting OOM, depending on your machine (GPU memory capacty), please also adjust the batch_size and max_length parameters accordingly.
> 
> **2.** For 8bit models, it seems some CPU will not support the mixed precision computation. We recommend you use GPU instead of CPU for running the code.
> 
> **3.** If you plan to use the fine-tuned LLM checkpoint for graph reasoning, please (1) replace the fine-tuned checkpoints in the downloaded **Graph_Toolformer_Package/koala/language_models/gptj_8bit/local_data/finetuned_model/graph_toolformer_GPTJ** directory of the Graph_Toolformer_Package for graph reasoning demos, (2) also remember to change the checkpoint names to **graph_toolformer_GPTJ** before pasting it to the koala folder, so the demo framework will load the checkpoint.

```
python3 gtoolformer_gptj_script.py
```

For more information, you can also refer to the [README.md provided in the LLM_Tuning directory](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/LLM_Tuning) as well.

### Play with Graph_Toolformer Demo code

After downloading the Graph_Toolformer_Package directory, also downloading the [graph_datasets.zip](https://drive.google.com/file/d/1lC23j9RYMb44JRJybxIpUtxuQ2lW58n_/view?usp=sharing), [koala/graph_models.zip](https://drive.google.com/file/d/1Wpw2DBa2p_PG_od3AVGZd5GvoKhCM2mH/view?usp=share_link) and [koala/language_models.zip](https://drive.google.com/file/d/1wja3_aBbWg6YztB7gNna_2FuN10dWjiZ/view?usp=sharing), installing the [conda environment](https://github.com/jwzhanggy/Graph_Toolformer/blob/main/environment.yml), you can just go ahead to run the code with the following command to start the demo

```
python3 ./src/Framework_Graph_Toolformer.py
```
You can type in inputs which are similar to the prompt inputs, and the model will carry out the reasoning task and return the otuputs. The reasoning process will call both the LLMs and GNN models, so generating the output will take some time. The GNN models are pre-trained according to the previous papers, and the LLMs is fine-tuned with the code in the LLM_Tunign directory based on the prompt datasets.

By changing the "if 0" to "if 1" in the bottom main function of Framework_Graph_Toolformer.py, you can try different reasoning tasks.

For more information, you can also refer to the [README.md provided in the Graph_Toolformer_Package directory](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/Graph_Toolformer_Package) as well.

************************************************************************************************

🔴
🟠
⚫
⚪
🟣
🟢
🟡
🔵


## Tasks to be done

- [x] 🟢 Polish the framework: 7/7 done
  - [x] 🟢 add working memory module
  - [x] 🟢 add query parser module
  - [x] 🟢 add query excutor module
  - [x] 🟢 add graph dataset hub
  - [x] 🟢 add graph model hub
  - [x] 🟢 add graph reasoning task hub
  - [x] 🟢 add llm model hub
- [x] 🟢 Expand the framework: 3/3 done
  - [x] 🟢 Include graph datasets: done
    - [x] 🟢 graph property dataset
    - [x] 🟢 bibliographic networks: cora, pubmed, citeseer
    - [x] 🟢 molecular graphs: proteins, nci1, mutag, ptc
    - [x] 🟢 social networks: twitter, foursquare
    - [x] 🟢 recommender system: amazon, last.fm, movielens  
    - [x] 🟢 knowledge graphs: wordnet, freebase 
  - [x] 🟢 Add pre-trained graph models: done
    - [x] 🟢 Toolx
    - [x] 🟢 Graph-Bert
    - [x] 🟢 SEG-Bert
    - [x] 🟢 KMeans Clustering
    - [x] 🟢 BPR
    - [x] 🟢 TransE
  - [x] 🟢 Include graph reasoning tasks: done
    - [x] 🟢 graph property reasoning
    - [x] 🟢 bibliographic paper topic reasoning
    - [x] 🟢 molecular graph function reasoning
    - [x] 🟢 social network community reasoning
    - [x] 🟢 recommender system reasoning
    - [x] 🟢 knowledge graph reasoning
- [x] 🟢 Polish and release the datasets: 4/4 released
  - [x] 🟢 graph raw data: 
  - [x] 🟢 graph reasoning prompt data
  - [x] 🟢 pre-trained graph model checkpoints
  - [x] 🟢 fine-tuned llm model checkpoints
    
- [ ] 🟠 Add and test more LLMs: 1/5 done
  - [x] 🟢 GPT-J
  - [ ] LLaMA
  - [ ] GPT-2
  - [ ] OPT
  - [ ] Bloom
 
- [ ] 🟠 Release the framework and service: 0/5 done
  - [ ] 🟠 Implement the CLI for framework usage
  - [ ] 🟠 Provide the demo for graph reasoning
  - [ ] 🟠 Add API for customerized graph reasoning
  - [ ] 🟠 Release GUI and web access/service


