'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class Prompt_Dataset(Dataset):
    def __init__(self, txt, labels, full, result):
        super().__init__()
        self.labels = labels
        self.text = txt
        self.full = full
        self.result = result

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        output = self.labels[idx]
        input = self.text[idx]
        full = self.full[idx]
        result = self.result[idx]
        sample = {"inputs": input, "outputs": output, "full": full, 'local_data': result}
        return sample

class Dataset_Prompt_Loader(dataset):

    data = None
    batch_size = None
    random_seed = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_prompts(self, file_path):
        f = open(file_path, 'r')
        input_data = []
        output_data = []
        full_data = []
        result_data = []
        for line in f:
            try:
                if "generated_data_loading_prompt_data_" in line:
                    line, _ = line.split("generated_data_loading_prompt_data_")
                    line += " Reasoning Result: None.\n"
                line = line.rstrip('.\n')
                input_str, remaining = line.split(" Output: ")
                input_data.append(input_str)
                output_str, _ = remaining.split(" Reasoning Result: ")
                output_str = "Output: " + output_str.rstrip()
                output_data.append(output_str)

                _, result = line.split(" Reasoning Result: ")
                full = input_str + ' ' + output_str
                full_data.append(full)
                result = "Reasoning Result: " + result
                result_data.append(result)
            except Exception as e:
                print(e, 'exception cases:', line)
        return input_data, output_data, full_data, result_data

    def load(self):
        print('prepare text graph_datasets...')
        train_input, train_output, train_full, train_result = self.load_prompts(
            self.dataset_source_folder_path+self.dataset_source_file_name+'_train')
        test_input, test_output, test_full, test_result = self.load_prompts(
            self.dataset_source_folder_path + self.dataset_source_file_name + '_test')

        train_dataset = Prompt_Dataset(train_input, train_output, train_full, train_result)
        test_dataset = Prompt_Dataset(test_input, test_output, test_full, test_result)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return {'train': train_dataloader, 'test': test_dataloader}