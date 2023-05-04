# Graph-Toolformer (project done)

> **Note**
<span style="color:red">May 4, 2023: All the datasets and model checkpoints have been publicized</span><br>
> - **Datasets and Model Checkpoints**: [See this page](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/data)
> - **LLMs Fine-Tuning with Prompts**: [See this page](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/LLM_Tuning)
> - **Graph-toolformer Demp**: [See this page](https://github.com/jwzhanggy/Graph_Toolformer/tree/main/Graph_Toolformer_Package)

************************************************************************************************

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
For the packages cannot be installed with the above conda command, you may consider to install via pip instead.

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
- [ ] 🟢 Expand the framework: 3/4 done
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


