import abc
import warnings
from collections import UserDict

class Hub_Graph_Task(UserDict):

    def __init__(self):
        UserDict.__init__(self)

    # a set of tasks to be studied in this project
    # more tasks (with different models and datasets) can be added
    def prepare(self, graph_data_hub, graph_model_hub):
        #-------------------------------
        self['graph_basic_property_reasoning'] = Graph_Basic_Property_Reasoning()
        # -------------------------------
        self['bibliographic_paper_topic_reasoning'] = Bibliographic_Network_Topic_Reasoning()
        # -------------------------------
        self['molecular_graph_function_reasoning'] = Molecular_Graph_Function_Reasoning()
        # -------------------------------
        self['recommender_system_reasoning'] = Recommender_System_Recommendation_Reasoning()
        # -------------------------------
        self['social_network_community_reasoning'] = Social_Network_Community_Reasoning()
        # -------------------------------
        self['knowledge_graph_entity_relation_reasoning'] = Knowledge_Graph_Searching_Reasoning()
        # -------------------------------

class Graph_Reasoning_Task():
    default_graph_model = None

    def __init__(self, name=None, description=None, default_graph_model=None):
        self.task_name = name
        self.task_description = description
        if default_graph_model is not None:
            self.default_graph_model = default_graph_model

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass

#================== fundamental graph reasoning tasks ======================
# all application graph tasks can be reduced to these fundamental tasks

from koala.graph_models.toolx.Toolx import Toolx
class Graph_Property(Graph_Reasoning_Task):

    def __init__(self, name=None, description=None, default_graph_model=None):
        Graph_Reasoning_Task.__init__(self, name=name, description=description, default_graph_model=default_graph_model)
        if self.default_graph_model is None:
            self.default_graph_model = Toolx()

    @staticmethod
    def run(self, api_function, graph_model=None, graph_data=None, extra_parameter=None):
        if graph_data is None:
            raise ValueError("The graph dataset in this reasoning task is missing.")
        if graph_model is None:
            graph_model = self.default_graph_model
            warnings.warn("No graph model is specified, the property reasoning task will use the default graph model {}".format(graph_model.model_name))
        if 'graph_set' in graph_data:
            attribute_list = []
            for graph in graph_data['graph_set']:
                attribute_list.append(graph_model.attribute_calculation(api_function, graph_data['graph_set'][graph]))
            if len(attribute_list) == 1:
                return attribute_list[0]
            else:
                return attribute_list
        elif 'nodes' in graph_data and 'links' in graph_data:
            return graph_model.attribute_calculation(api_function, graph_data)


from koala.graph_models.graph_bert.GraphBertNodeClassification import GraphBertNodeClassification as Graph_Bert
class Node_Classification(Graph_Reasoning_Task):

    def __init__(self, name=None, description=None, default_graph_model=None):
        Graph_Reasoning_Task.__init__(self, name=name, description=description, default_graph_model=default_graph_model)
        if self.default_graph_model is None:
            self.default_graph_model = Graph_Bert()

    @staticmethod
    def run(self, api_function, graph_model=None, graph_data=None, extra_parameter=None):
        if graph_data is None:
            raise ValueError("The graph dataset in this reasoning task is missing.")
        if graph_model is None:
            graph_model = Graph_Bert(graph_name=graph_data['data_profile']['name'])
            warnings.warn("No graph model is specified, the task will use the default graph model {}".format(graph_model.model_name))

        if api_function == 'node-classification':
            return graph_model.node_classification(graph_data, extra_parameter)
        else:
            raise ValueError("Unrecognized graph reasoning task {}.".format(api_function))

from koala.graph_models.seg_bert.SegmentedGraphBertGraphClassification import SegmentedGraphBertGraphClassification as SEG_Bert
class Graph_Clasisfication(Graph_Reasoning_Task):

    def __init__(self, name=None, description=None, default_graph_model=None):
        Graph_Reasoning_Task.__init__(self, name=name, description=description, default_graph_model=default_graph_model)
        if self.default_graph_model is None:
            self.default_graph_model = SEG_Bert()

    @staticmethod
    def run(self, api_function, graph_model=None, graph_data=None, extra_parameter=None):
        if graph_data is None:
            raise ValueError("The graph dataset in this reasoning task is missing.")
        if graph_model is None:
            graph_model = self.default_graph_model
            warnings.warn("No graph model is specified, the task will use the default graph model {}".format(graph_model.model_name))

        if api_function == 'graph-classification':
            return graph_model.graph_classification(graph_data, extra_parameter)
        else:
            raise ValueError("Unrecognized graph reasoning task {}.".format(api_function))

from koala.graph_models.bpr.BPR import BPR
class Graph_Ranking(Graph_Reasoning_Task):

    def __init__(self, name=None, description=None, default_graph_model=None):
        Graph_Reasoning_Task.__init__(self, name=name, description=description, default_graph_model=default_graph_model)
        if self.default_graph_model is None:
            self.default_graph_model = BPR()

    @staticmethod
    def run(self, api_function, graph_model=None, graph_data=None, extra_parameter=None):
        if graph_data is None:
            raise ValueError("The graph dataset in this reasoning task is missing.")
        if graph_model is None:
            graph_model = self.default_graph_model
            warnings.warn("No graph model is specified, the task will use the default graph model {}".format(graph_model.model_name))

        if api_function == 'recommendation':
            return graph_model.recommendation(graph_data, extra_parameter)
        elif api_function == 'topk_recommendation':
            return graph_model.topk_recommendation(graph_data, extra_parameter)
        else:
            raise ValueError("Unrecognized graph reasoning task {}.".format(api_function))


from koala.graph_models.kmeans.KMeans import KMeans
class Graph_Clustering(Graph_Reasoning_Task):

    def __init__(self, name=None, description=None, default_graph_model=None):
        Graph_Reasoning_Task.__init__(self, name=name, description=description, default_graph_model=default_graph_model)
        if self.default_graph_model is None:
            self.default_graph_model = KMeans()

    @staticmethod
    def run(self, api_function, graph_model=None, graph_data=None, extra_parameter=None):
        if graph_data is None:
            raise ValueError("The graph dataset in this reasoning task is missing.")
        if graph_model is None:
            graph_model = self.default_graph_model
            warnings.warn("No graph model is specified, the task will use the default graph model {}".format(graph_model.model_name))

        if api_function == 'community_avg_size':
            return graph_model.community_avg_size(graph_data)
        elif api_function == 'community_max_size':
            return graph_model.community_max_size(graph_data)
        elif api_function == 'community_count':
            return graph_model.community_count(graph_data)
        elif api_function == 'common_community_check':
            boolean_tag = graph_model.common_community_check(graph_data, extra_parameter)
            if boolean_tag:
                return 'the same'
            else:
                return 'different'
        elif api_function == 'community':
            return graph_model.community(graph_data, extra_parameter)
        else:
            raise ValueError("Unrecognized graph reasoning task {}.".format(api_function))

        
from koala.graph_models.transe.TransE import TransE
class Graph_Searching(Graph_Reasoning_Task):

    def __init__(self, name=None, description=None, default_graph_model=None):
        Graph_Reasoning_Task.__init__(self, name=name, description=description, default_graph_model=default_graph_model)
        if self.default_graph_model is None:
            self.default_graph_model = TransE()

    @staticmethod
    def run(self, api_function, graph_model=None, graph_data=None, extra_parameter=None):
        if graph_data is None:
            raise ValueError("The graph dataset in this reasoning task is missing.")
        if graph_model is None:
            graph_model = self.default_graph_model
            warnings.warn("No graph model is specified, the task will use the default graph model {}".format(graph_model.model_name))
        if extra_parameter is None:
            raise ValueError("The parameters for this searching task is missing.")

        if api_function == 'tail_entity':
            return graph_model.search_tail_entity(graph_data, extra_parameter)
        elif api_function == 'head_entity':
            return graph_model.search_head_entity(graph_data, extra_parameter)
        elif api_function == 'relation':
            return graph_model.search_relation(graph_data, extra_parameter)
        else:
            raise ValueError("Unrecognized graph searching task {}.".format(api_function))


#--------------------- applied graph reasoning tasks ------------------------------
# these tasks can be reduced to the fundamental graph reasoning tasks defined above
# for some tasks that cannot be reduced to the above tasks, they can also be added below
# remember to define the customerized run() function to handle the reasoning queries

class Graph_Basic_Property_Reasoning(Graph_Property):
    def __init__(self, *args, **kwargs):
        Graph_Property.__init__(self, *args, **kwargs)

    def run(self, api_function, *args, **kwargs):
        if api_function in ['order', 'size', 'density', 'eccentricity', 'radius', 'center', 'shortest_path',
                            'avg_path_length', 'min_path_length', 'max_path_length', 'periphery', 'diameter']:
            return Graph_Property.run(self, api_function, *args, **kwargs)
        else:
            raise ValueError("Unrecognized graph property reasoning objective {}.".format(api_function))


class Bibliographic_Network_Topic_Reasoning(Node_Classification):
    def __init__(self, *args, **kwargs):
        Node_Classification.__init__(self, *args, **kwargs)

    def run(self, api_function, *args, **kwargs):
        if api_function == 'topic':
            if 'extra_parameter' in kwargs and kwargs['extra_parameter'] is not None:
                updated_extra_parameter = []
                for paper_id in kwargs['extra_parameter']:
                    _, updated_paper_id = paper_id.split('#')
                    updated_extra_parameter.append(int(updated_paper_id))
                kwargs['extra_parameter'] = updated_extra_parameter
            return Node_Classification.run(self, 'node-classification', *args, **kwargs)
        else:
            raise ValueError("Unrecognized bibliographic network reasoning task {}.".format(api_function))

class Molecular_Graph_Function_Reasoning(Graph_Clasisfication):
    def __init__(self, *args, **kwargs):
        Graph_Clasisfication.__init__(self, *args, **kwargs)

    def run(self, api_function, *args, **kwargs):
        if api_function == 'molecule_function':
            if 'extra_parameter' in kwargs and kwargs['extra_parameter'] is not None:
                updated_extra_parameter = []
                for graph_id in kwargs['extra_parameter']:
                    _, updated_graph_id = graph_id.split('#')
                    updated_extra_parameter.append(int(updated_graph_id))
                kwargs['extra_parameter'] = updated_extra_parameter
            return Graph_Clasisfication.run(self, 'graph-classification', *args, **kwargs)
        else:
            raise ValueError("Unrecognized molecular graph reasoning task {}.".format(api_function))

class Social_Network_Community_Reasoning(Graph_Clustering):
    def __init__(self, *args, **kwargs):
        Graph_Clustering.__init__(self, *args, **kwargs)

    def run(self, api_function, *args, **kwargs):
        if api_function in ['community', 'community_avg_size', 'community_max_size',
                            'community_count', 'common_community_check']:
            if 'extra_parameter' in kwargs and kwargs['extra_parameter'] is not None:
                updated_extra_parameter = []
                if len(kwargs['extra_parameter']) == 1 and '{' in kwargs['extra_parameter'][0] and '}' in kwargs['extra_parameter'][0]:
                    user_list_str = kwargs['extra_parameter'][0].replace('{', '').replace('}', '').split(',')
                else:
                    user_list_str = kwargs['extra_parameter']
                for user_id in user_list_str:
                    _, updated_user_id = user_id.split('#')
                    updated_extra_parameter.append(updated_user_id)
                kwargs['extra_parameter'] = updated_extra_parameter
            return Graph_Clustering.run(self, api_function, *args, **kwargs)
        else:
            raise ValueError("Unrecognized social network reasoning task {}.".format(api_function))

class Recommender_System_Recommendation_Reasoning(Graph_Ranking):
    def __init__(self, *args, **kwargs):
        Graph_Ranking.__init__(self, *args, **kwargs)

    def run(self, api_function, *args, **kwargs):
        if api_function in ['topk_recommendation', 'recommendation']:
            if 'extra_parameter' in kwargs and kwargs['extra_parameter'] is not None:
                updated_extra_parameter = []
                for parameter in kwargs['extra_parameter']:
                    if '#' in parameter:
                        _, updated_parameter = parameter.split('#')
                        updated_extra_parameter.append(updated_parameter)
                    elif parameter.isdigit():
                        updated_parameter = int(parameter)
                        updated_extra_parameter.append(updated_parameter)
                kwargs['extra_parameter'] = updated_extra_parameter
            return Graph_Ranking.run(self, api_function, *args, **kwargs)
        else:
            raise ValueError("Unrecognized recommender system reasoning task {}.".format(api_function))

class Knowledge_Graph_Searching_Reasoning(Graph_Searching):
    def __init__(self, *args, **kwargs):
        Graph_Searching.__init__(self, *args, **kwargs)

    def run(self, api_function, *args, **kwargs):
        if api_function in ['head_entity', 'relation', 'tail_entity']:
            if 'extra_parameter' in kwargs and kwargs['extra_parameter'] is not None:
                updated_extra_parameter = []
                for parameter in kwargs['extra_parameter']:
                    if '#' in parameter:
                        _, updated_parameter = parameter.split('#')
                        updated_extra_parameter.append(updated_parameter)
                    elif parameter.isdigit():
                        updated_parameter = int(parameter)
                        updated_extra_parameter.append(updated_parameter)
                kwargs['extra_parameter'] = updated_extra_parameter
            return Graph_Searching.run(self, api_function, *args, **kwargs)
        else:
            raise ValueError("Unrecognized knowledge graph reasoning task {}.".format(api_function))
