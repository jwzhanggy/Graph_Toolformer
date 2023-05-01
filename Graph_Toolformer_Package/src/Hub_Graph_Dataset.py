
import pickle
from collections import UserDict

class Hub_Graph_Dataset(UserDict):
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self):
        UserDict.__init__(self)

    # a set of graph benchmark datasets to be used for this project
    # more graph datasets can be added later
    def prepare(self):
        # ---- graph_properties ----
        self['gpr'] = Graph_Dataset_Loader(
            name='gpr', description='graph property reasoning dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='gpr')
        #---- bibliographic_networks ----
        self['cora'] = Graph_Dataset_Loader(
            name='cora', description='cora bibliographic_network dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='cora')
        self['pubmed'] = Graph_Dataset_Loader(
            name='pubmed', description='pubmed bibliographic_network dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='pubmed')
        self['citeseer'] = Graph_Dataset_Loader(
            name='citeseer', description='citeseer bibliographic_network dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='citeseer')
        # ---- molecular_graphs ----
        self['mutag'] = Graph_Dataset_Loader(
            name='mutag', description='mutag molecular graph dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='mutag')
        self['nci1'] = Graph_Dataset_Loader(
            name='nci1', description='nci1 molecular graph dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='nci1')
        self['proteins'] = Graph_Dataset_Loader(
            name='proteins', description='proteins molecular graph dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='proteins')
        self['ptc'] = Graph_Dataset_Loader(
            name='ptc', description='ptc molecular graph dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='ptc')
        # ---- recommender_systems ----
        self['amazon'] = Graph_Dataset_Loader(
            name='amazon', description='amazon recommender system dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='amazon')
        self['last-fm'] = Graph_Dataset_Loader(
            name='last-fm', description='last-fm recommender system dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='last-fm')
        self['movielens'] = Graph_Dataset_Loader(
            name='movielens', description='movielens recommender system dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='movielens')
        # ---- social_networks ----
        self['twitter'] = Graph_Dataset_Loader(
            name='twitter', description='twitter social network dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='twitter')
        self['foursquare'] = Graph_Dataset_Loader(
            name='foursquare', description='foursquare social network dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='foursquare')
        # ---- knowledge_graphs ----
        self['freebase'] = Graph_Dataset_Loader(
            name='freebase', description='freebase knowledge graph dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='freebase')
        self['worldnet'] = Graph_Dataset_Loader(
            name='worldnet', description='worldnet knowledge graph dataset',
            dataset_source_folder_path='./graph_datasets/',
            dataset_source_file_name='worldnet')

    def get_dataset_name(self, name_str):
        pass

class Graph_Dataset_Loader:
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, name=None, description=None, dataset_source_folder_path=None, dataset_source_file_name=None,):
        self.name = name
        self.description = description
        self.dataset_source_folder_path = dataset_source_folder_path
        self.dataset_source_file_name = dataset_source_file_name

    def load(self, **kwargs):
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        raw_graph_data = pickle.load(f)
        f.close()
        if "graph_set" in raw_graph_data:
            return self.filter_graph_set(raw_graph_data=raw_graph_data, **kwargs)
        else:
            return self.filter_graph(raw_graph_data=raw_graph_data, **kwargs)

    def filter_graph_set(self, raw_graph_data, **kwargs):
        try:
            graph_instance_subset = kwargs['filter_parameter'][0]
        except Exception as e:
            graph_instance_subset = None
        if graph_instance_subset is not None and (' ' in graph_instance_subset or '"' in graph_instance_subset or '{' in graph_instance_subset or '}' in graph_instance_subset):
            graph_instance_subset = graph_instance_subset.replace(' ', '').replace('"', '').replace('{', '').replace('}', '').split(',')
        if graph_instance_subset is None or type(graph_instance_subset) == str and (graph_instance_subset == 'all graph instances' or graph_instance_subset.startswith('all')):
            return raw_graph_data
        elif type(graph_instance_subset) in [list, tuple, dict, set]:
            filtered_graph_data = {
                "data_profile": raw_graph_data["data_profile"],
                "graph_set": {}
            }
            for graph_instance in graph_instance_subset:
                filtered_graph_data["graph_set"][graph_instance] = raw_graph_data["graph_set"][graph_instance]
            return filtered_graph_data

    def filter_graph(self, raw_graph_data, **kwargs):
        try:
            node_subset = kwargs['filter_parameter'][0][0]
        except Exception as e:
            node_subset = None
        try:
            link_subset = kwargs['filter_parameter'][0][1]
        except Exception as e:
            link_subset = None

        if (node_subset is None and link_subset is None) or ((node_subset == 'all nodes' or node_subset.startswith('all related')) and (link_subset == 'all links' or link_subset.startswith('all related'))):
            return raw_graph_data
        else:
            filtered_graph_data = {
                "data_profile": raw_graph_data["data_profile"],
                "node": {},
                "link": {},
            }
            if type(node_subset) in ['list', 'tuple', 'dict', 'set']:
                if type(link_subset) in ['str'] and (link_subset == 'all links' or link_subset.startswith('all related')):
                    for node in node_subset:
                        filtered_graph_data["node"][node] = raw_graph_data["node"][node]
                        if node not in filtered_graph_data["link"]:
                            filtered_graph_data["link"][node] = {}
                        for connected_node in raw_graph_data["link"][node]:
                            filtered_graph_data["link"][node][connected_node] = raw_graph_data["link"][node][connected_node]
                    return filtered_graph_data
                elif type(link_subset) in ['list', 'tuple', 'dict', 'set']:
                    for node in node_subset:
                        filtered_graph_data["node"][node] = raw_graph_data["node"][node]
                    for node_pair in link_subset:
                        node, connected_node = node_pair
                        if node not in filtered_graph_data["link"]:
                            filtered_graph_data["link"][node] = {}
                        filtered_graph_data["link"][node][connected_node] = raw_graph_data["link"][node][connected_node]
                        if raw_graph_data["graph_data_information"]["is_directed"] == False:
                            if connected_node not in filtered_graph_data["link"]:
                                filtered_graph_data["link"][connected_node] = {}
                            filtered_graph_data["link"][connected_node][node] = raw_graph_data["link"][connected_node][node]
                    return filtered_graph_data
            elif type(link_subset) in ['list', 'tuple', 'dict', 'set']:
                if type(node_subset) in ['str']:
                    for node_pair in link_subset:
                        node, connected_node = node_pair
                        filtered_graph_data["node"][node] = raw_graph_data["node"][node]
                        filtered_graph_data["node"][connected_node] = raw_graph_data["node"][connected_node]
                        if node not in filtered_graph_data["link"]:
                            filtered_graph_data["link"][node] = {}
                        filtered_graph_data["link"][node][connected_node] = raw_graph_data["link"][node][connected_node]
                        if raw_graph_data["graph_data_information"]["is_directed"] == False:
                            if connected_node not in filtered_graph_data["link"]:
                                filtered_graph_data["link"][connected_node] = {}
                            filtered_graph_data["link"][connected_node][node] = raw_graph_data["link"][connected_node][node]
                    return filtered_graph_data


if __name__ == "__main__":
    data_hub = Hub_Graph_Dataset()
    data_hub.prepare()
    print(data_hub['cora'].load()['data_profile'])