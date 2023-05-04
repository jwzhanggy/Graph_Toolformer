# Source code for Graph-Toolformer LLM Fine-Tuning with graph reasoning prompts

![pipeline!](pipeline.png)

************************************************************************************************

## What this directory is used for?

In this directory, we will fine-tune the LLMs with graph reasoning prompt datasets. The tuned LLMs will be copied to the Graph_Toolformer_Package (the fine-tuned LLMs in the koala directory) for various graph reasoning tasks.

************************************************************************************************

## How to run the code?

After downloading the LLM_Tuning directory, installing the conda environment ([**see this file**](https://github.com/jwzhanggy/Graph_Toolformer/blob/main/environment.yml)), you can just go ahead to run the code with the following command

```
python3 gtoolformer_gptj_script.py
```

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
