import dgl
from augmenter.sync_graph_data import SyncGraphData
from augmenter.gcnnorm import GCNNorm


class Merge(SyncGraphData):
    def __init__(self, args, data, device):
        super().__init__(args, data)
        self.device = device

    def _augment(self, dataset):
        merged_graphs = [dataset[0]]

        for graph in dataset[1:]:
            merged_graph = dgl.merge([merged_graphs[-1], graph])
            merged_graph = merged_graph.cpu().to_simple().to(self.device)
            del merged_graph.edata["count"]
            merged_graphs.append(merged_graph)

        return GCNNorm(self.device)(merged_graphs)
