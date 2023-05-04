from Module_Query_Parser import Module_Query_Parser
from Module_Working_Memory import Module_Working_Memory
from Module_LLM_Trainer import Module_LLM_Trainer
from Hub_Graph_Dataset import Hub_Graph_Dataset
from Hub_Graph_Model import Hub_Graph_Model
from Hub_Graph_Task import Hub_Graph_Task
from Hub_Language_Model import Hub_Language_Model
from Config_Graph_Toolformer import Config_Graph_Toolformer

import re
import torch

class Framework_Graph_Toolformer:
    selected_language_model = None

    def __init__(self, name='', description=''):
        self.name = name
        self.description = description
        self.config = Config_Graph_Toolformer()

        self.working_memory = Module_Working_Memory(maxlen=self.config.framework_config['work_memory_maxlen'])
        self.query_parser = Module_Query_Parser()

        self.graph_model_hub = Hub_Graph_Model()
        self.graph_data_hub = Hub_Graph_Dataset()
        self.graph_task_hub = Hub_Graph_Task()
        self.language_model_hub = Hub_Language_Model()
        self.selected_language_model = 'gptj-8bit'

    def prepare_setting(self):

        self.working_memory.prepare()
        self.graph_data_hub.prepare()
        self.graph_model_hub.prepare()
        self.graph_task_hub.prepare(graph_data_hub=self.graph_data_hub, graph_model_hub=self.graph_model_hub)

        hyper_parameter_dict = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_length': 128,
            'special_token_dict': {'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>', 'pad_token': '<'},
        }
        self.language_model_hub.prepare(hyper_parameters=hyper_parameter_dict)

    def statement_updating(self, statement, result_dict):
        output_statement = statement
        for query_str in result_dict:
            if result_dict[query_str][0]:
                output_statement = output_statement.replace(query_str, str(result_dict[query_str][1]))
            else:
                output_statement = output_statement.replace('{}'.format(query_str), '')
        output_statement = output_statement.replace('  ', ' ')
        return output_statement

    def query_extraction(self, statement):
        pattern = r'\[(.*?)\]'
        detected_query_list = re.findall(pattern, statement)
        query_dict = {}
        for query_str in detected_query_list:
            query_str = '[{}]'.format(query_str)
            query_dict[query_str] = self.query_parser.parse_query(query_str)
        return query_dict

    def query_execution(self, query_dict):
        result_dict = {}
        for query_str in query_dict:
            query, output_tag = query_dict[query_str]
            query_memory_key = query_str.replace(' ', '').lower()
            if query_memory_key in self.working_memory:
                result = self.working_memory[query_memory_key]
            else:
                result = self.api_calling(query)
                self.working_memory[query_memory_key] = result
            result_dict[query_str] = (output_tag[0], result)
        return result_dict

    def api_calling(self, query):
        query_type = query[0]
        parameter_list = []
        for parameter in query[1]:
            if isinstance(parameter, str): # normal parameters
                parameter_list.append(parameter)
            else: # nested api calls
                parameter_list.append(self.api_calling(parameter))
        if query_type == 'GR':
            return self.graph_reasoning(parameter_list)
        elif query_type == 'GL':
            return self.graph_loading(parameter_list)
        else:
            raise ValueError('graph reasoning API call {} cannot be recognized, consider to change it to GR or GL.'.format(query_type))

    def graph_loading(self, parameter_list):
        query_memory_key = 'GL({})'.format(','.join(parameter_list)).replace(' ', '').lower()

        if query_memory_key in self.working_memory:
            return self.working_memory[query_memory_key]

        # the parameter contains the graph graph_datasets name
        elif len(parameter_list) == 1:
            dataset_name = parameter_list[0]
            graph_data = self.graph_data_hub[dataset_name].load()
            self.working_memory[query_memory_key] = graph_data
            return self.working_memory[query_memory_key]
        # the parameter contains graph graph_datasets name, node subset, link subset
        elif len(parameter_list) <= 3:
            dataset_name = parameter_list[0]
            graph_data = self.graph_data_hub[dataset_name].load(filter_parameter=parameter_list[1:])
            self.working_memory[query_memory_key] = graph_data
            return self.working_memory[query_memory_key]
        else:
            raise ValueError('graph loading input parameter {} cannot be parsed.'.format(','.join(parameter_list)))

    def graph_reasoning(self, parameter_list):
        graph_data = parameter_list[0]
        api_call = parameter_list[1]
        if len(parameter_list) > 2:
            extra_parameter = parameter_list[2:]
        else:
            extra_parameter = None
        if ':' in api_call:
            model, api_function = api_call.split(':')
        else:
            model = None
            api_function = api_call

        if model is not None and model in self.graph_model_hub:
            model = self.graph_model_hub[model]

        if api_function in ["order", "size", "radius", "diameter", "eccentricity", "center", "periphery", "density",
                            "min_path_length", "max_path_length", "avg_path_length", "shortest_path"]:
            return self.graph_task_hub['graph_basic_property_reasoning'].run(api_function=api_function,
                                                                             graph_model=model,
                                                                             graph_data=graph_data,
                                                                             extra_parameter=extra_parameter)
        elif api_function in ['topic']:
            return self.graph_task_hub['bibliographic_paper_topic_reasoning'].run(api_function=api_function,
                                                                                  graph_model=model,
                                                                                  graph_data=graph_data,
                                                                                  extra_parameter=extra_parameter)
        elif api_function in ['molecule_function']:
            return self.graph_task_hub['molecular_graph_function_reasoning'].run(api_function=api_function,
                                                                                 graph_model=model,
                                                                                 graph_data=graph_data,
                                                                                 extra_parameter=extra_parameter)
        elif api_function in ['recommendation', 'topk_recommendation']:
            return self.graph_task_hub['recommender_system_reasoning'].run(api_function=api_function,
                                                                           graph_model=model,
                                                                           graph_data=graph_data,
                                                                           extra_parameter=extra_parameter)
        elif api_function in ['community', 'community_avg_size', 'community_max_size',
                              'community_count', 'common_community_check']:
            return self.graph_task_hub['social_network_community_reasoning'].run(api_function=api_function,
                                                                                 graph_model=model,
                                                                                 graph_data=graph_data,
                                                                                 extra_parameter=extra_parameter)
        elif api_function in ['head_entity', 'relation', 'tail_entity']:
            return self.graph_task_hub['knowledge_graph_entity_relation_reasoning'].run(api_function=api_function,
                                                                                        graph_model=model,
                                                                                        graph_data=graph_data,
                                                                                        extra_parameter=extra_parameter)
        else:
            raise ValueError('unrecognized pretrained_model:function {}:{}, consider to expand the graph pretrained_model hub to handle the desired graph reasoning tasks.'.format(model, api_function))

    def inference(self, input_str=None):
        if input_str is None or input_str == '':
            raise ValueError("The input of the inference function cannot be None or blank")
        if 'Input: ' in input_str:
            input_str = input_str.lstrip('Input: ')
        payload = 'Input: ' + input_str + ' Output: '
        return self.language_model_hub[self.selected_language_model].inference(input_text=payload)

    def run(self):
        def print_in_color(text):
            red_color = "\033[96m"
            reset_color = "\033[0m"
            print(f"{red_color}{text}{reset_color}")

        while True:
            try:
                input_str = input("Input (press Enter to stop): ")
                if input_str == '': break
                generation_output = self.inference(input_str=input_str)
                _, generation_output = generation_output[0].split(' Output: ')
                query_dict = panel.query_extraction(generation_output)
                result_dict = panel.query_execution(query_dict)
                output_statement = panel.statement_updating(generation_output, result_dict)
                print_in_color(output_statement)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    print('='*70)

    if 1:
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        panel.run()

    if 0:
        input_str = 'Input: The radius of the truncated tetrahedron graph is the minimum eccentricity of any node, which in this case is [TBR].'
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        print(panel.inference(input_str=input_str))

    if 0:
        # property reasoning testing
        generation_output = 'Output: \nThe radius of the truncated tetrahedron graph is the minimum eccentricity of any node, which in this case is [GR(GL("gpr", {"truncated_tetrahedron_graph"}), "toolx:eccentricity")-->r].'
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        query_dict = panel.query_extraction(generation_output)
        result_dict = panel.query_execution(query_dict)
        output_statement = panel.statement_updating(generation_output, result_dict)
        print(output_statement)
    if 0:
        # bibliographic network testing
        generation_output = 'Output: The topic of paper #31336 in the cora bibliographic network is [GR(GL("cora"), "graph_bert:topic", paper#31336)-->r]. Reasoning Result: Neural Networks.'
        print(generation_output)
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        query_dict = panel.query_extraction(generation_output)
        result_dict = panel.query_execution(query_dict)
        output_statement = panel.statement_updating(generation_output, result_dict)
        print(output_statement)
    if 0:
        # molecular graph testing
        generation_output = 'Output: The function for the chemical molecular graph #1724 in nci1 is [GR(GL("nci1"), "seg_bert:molecule_function", instance#1724)-->r]. Reasoning Result: 1.'
        print(generation_output)
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        query_dict = panel.query_extraction(generation_output)
        result_dict = panel.query_execution(query_dict)
        output_statement = panel.statement_updating(generation_output, result_dict)
        print(output_statement)

    if 0:
        # social network community testing
        generation_output = 'Output: In foursquare, the id of user 0mkarr\'s community is [GR(GL("foursquare"), "kmeans:community", {user#0mkarr, user#1000museums})-->r]. Reasoning Result: Unsupervised.'
        print(generation_output)
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        query_dict = panel.query_extraction(generation_output)
        result_dict = panel.query_execution(query_dict)
        output_statement = panel.statement_updating(generation_output, result_dict)
        print(output_statement)

    if 0:
        # recommender system testing

        generation_output = 'Output: The likelihood that user #A3QJU4FEN8PQSZ will be interested in item #B001FDG0IW in Amazon is [GR(GL("amazon"), "bpr:recommendation", user#A3QJU4FEN8PQSZ, item#B001FDG0IW)-->r].'
        print(generation_output)
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        query_dict = panel.query_extraction(generation_output)
        result_dict = panel.query_execution(query_dict)
        output_statement = panel.statement_updating(generation_output, result_dict)
        print(output_statement)

    if 0:
        # knowledge graph reasoning
        generation_output = 'Output: According to the Freebase knowledge graph, the relation between entity#/m/027rn and entity#/m/06cx9 is [GR(GL("freebase"), "transe:head_entity", relation#/location/country/form_of_government, entity#/m/06cx9)-->r]. Reasoning Result: /location/country/form_of_government.'
        print(generation_output)
        panel = Framework_Graph_Toolformer()
        panel.prepare_setting()
        query_dict = panel.query_extraction(generation_output)
        result_dict = panel.query_execution(query_dict)
        output_statement = panel.statement_updating(generation_output, result_dict)
        print(output_statement)
