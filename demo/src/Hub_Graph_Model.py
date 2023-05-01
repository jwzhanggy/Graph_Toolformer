from collections import UserDict

from koala.graph_models.toolx.Toolx import Toolx
from koala.graph_models.graph_bert.GraphBertNodeClassification import GraphBertNodeClassification as Graph_Bert
from koala.graph_models.seg_bert.SegmentedGraphBertGraphClassification import SegmentedGraphBertGraphClassification as SEG_Bert
from koala.graph_models.bpr.BPR import BPR
from koala.graph_models.kmeans.KMeans import KMeans
from koala.graph_models.transe.TransE import TransE


class Hub_Graph_Model(UserDict):

    def __init__(self):
        UserDict.__init__(self)

    # a set of basic graph models to be used for the project
    # more graph models can be added afterwards
    def prepare(self):
        self['toolx'] = Toolx()
        self['graph_bert'] = Graph_Bert()
        self['seg_bert'] = SEG_Bert()
        self['bpr'] = BPR()
        self['kmeans'] = KMeans()
        self['transe'] = TransE()
