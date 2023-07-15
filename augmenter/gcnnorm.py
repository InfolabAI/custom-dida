import utils_TiaRa
from augmenter.sync_graph_data import SyncGraphData


class GCNNorm(SyncGraphData):
    def __init__(self, args, data, device):
        super().__init__(args, data)
        self.device = device

    def _augment(self, dataset):
        normalized = [
            utils_TiaRa.graph_to_normalized_adjacency(graph) for graph in dataset
        ]
        return [utils_TiaRa.weighted_adjacency_to_graph(adj) for adj in normalized]
