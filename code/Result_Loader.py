'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.result import result
import pickle


class Result_Loader(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def load(self):
        #print('loading results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'rb')
        self.data = pickle.load(f)
        f.close()
        return self.data