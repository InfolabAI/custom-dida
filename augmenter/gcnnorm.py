import utils_TiaRa
from augmenter.augmenter import Augmenter


class GCNNorm(Augmenter):
    def __init__(self, data, device):
        super().__init__(data)
        self.device = device

    def _augment(self, dataset):
        normalized = [
            utils_TiaRa.graph_to_normalized_adjacency(graph) for graph in dataset
        ]
        return [utils_TiaRa.weighted_adjacency_to_graph(adj) for adj in normalized]
