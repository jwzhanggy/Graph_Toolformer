from code.Dataset_Prompt_Loader import Dataset_Prompt_Loader
from code.Method_Graph_Toolformer_GPTJ import Method_Graph_Toolformer_GPTJ
from code.Result_Saver_Loader import Result_Saver
from code.Setting_Regular import Setting_Regular
from code.Evaluate_Metrics import Evaluate_Metrics

import numpy as np
import torch
import time
import random

method_name_list = ['GPTJ']

method_checkpoint_cache_dict = {
    'GPTJ': {
        'checkpoint': 'EleutherAI/gpt-j-6b',
        'cache_dir': './pretrained_model/gpt-j-6b-8bit'
    },
}

task_data_dict = {
    'graph_data_loading': {None: 0},
    #'graph_properties': {None: 0},
    #'bibliographic_networks': {'cora': 0, 'citeseer': 0, 'pubmed': 0},
    #'molecular_graphs': {'mutag': 0, 'nci1': 0, 'proteins': 0, 'ptc': 0},
    #'recommender_systems': {'movielens': 0, 'last-fm': 0, 'amazon': 0},
    #'social_networks': {'dblp': 0, 'email': 0, 'youtube': 0},
    #'knowledge_graphs': {'freebase': 0, 'worldnet': 0},
}



for method_name in method_name_list:
    for reasoning_task in task_data_dict:
        for dataset_name in task_data_dict[reasoning_task]:
            #---- GPTJ fine-tuning and testing ----
            if 1:
                #---- parameter section -------------------------------
                random_seed = 2
                random.seed(random_seed)
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)
                # use a smaller batch_size and max_length for GPUs with smaller memory
                # 4090: batch_size 8, max_length 128 or batch_size 4, max_length 256
                # 1080Ti: batch_size 1 or 2, max_length 64
                batch_size = 8
                max_length = 128
                max_epoch = 12
                device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
                fine_tuned_checkpoint_filename = '{}_{}_{}_{}_{}'.format(method_name, reasoning_task, dataset_name, max_epoch, max_length)
                print(fine_tuned_checkpoint_filename)
                #------------------------------------------------------

                # ---- objection initialization setction ---------------
                data_obj = Dataset_Prompt_Loader('{} prompt: {}'.format(reasoning_task, dataset_name), '')
                data_obj.batch_size = batch_size
                data_obj.random_seed = random_seed
                data_obj.dataset_source_folder_path = './prompt/{}/'.format(reasoning_task)
                if dataset_name is None:
                   data_obj.dataset_source_file_name = 'prompts'
                else:
                   data_obj.dataset_source_file_name = '{}/prompts'.format(dataset_name)

                method_obj = Method_Graph_Toolformer_GPTJ(method_name, '', checkpoint_name=method_checkpoint_cache_dict[method_name]['checkpoint'], cache_dir=method_checkpoint_cache_dict[method_name]['cache_dir'], device=device)
                method_obj.fine_tuned_checkpoint_filename = fine_tuned_checkpoint_filename
                method_obj.max_length = max_length
                method_obj.max_epoch = max_epoch

                result_obj = Result_Saver('saver', '')
                result_obj.result_destination_folder_path = '../result/'
                result_obj.result_destination_file_name = 'Graph_Toolformer_{}_generation_result'.format(fine_tuned_checkpoint_filename)

                setting_obj = Setting_Regular('regular settings', '')

                evaluate_obj = Evaluate_Metrics('Metrics', '')
                # ------------------------------------------------------

                # ---- running section ---------------------------------
                print('************ {} Start@{} ************'.format(fine_tuned_checkpoint_filename, device))
                t_begin = time.time()
                setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
                setting_obj.print_setup_summary()
                setting_obj.load_run_save_evaluate()
                print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
                print('************ {} Finishsave_checkpoint ************'.format(fine_tuned_checkpoint_filename, dataset_name))
                # ------------------------------------------------------
