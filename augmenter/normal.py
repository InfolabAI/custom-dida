import torch, dgl
import utils_TiaRa
from tqdm import tqdm
from augmenter.sync_graph_data import SyncGraphData


class Normal(SyncGraphData):
    def __init__(self, args, data):
        super().__init__(args, data)

    def _augment(self, dataset):
        return dataset
