from convert_graph_types import ConvertGraphTypes
from tqdm import tqdm


class Augmenter:
    def __init__(self, num_nodes, original_edge_tensors):
        """
        Parameters
        --------
        num_nodes: int
        original_edge_tensors: [[2, num_edges], ...]
        """
        self.cgt = ConvertGraphTypes()
        self.list_of_dgl_graphs = []
        # tqdm with description
        for edge_tensor in tqdm(
            original_edge_tensors,
            desc="Converting the original edge tensors into DGL graphs...",
        ):
            nx_graph = self.cgt.edge_tensor_to_networkx(edge_tensor, num_nodes)
            self.list_of_dgl_graphs.append(self.cgt.networkx_to_dgl(nx_graph))

    def __call__(self):
        augmented_list_of_dgl_graphs = self._augment(self.list_of_dgl_graphs)
        list_of_augmented_edge_tensors = []
        for dgl_graph in tqdm(
            augmented_list_of_dgl_graphs,
            desc="Converting the augmented DGL graphs into edge tensors...",
        ):
            nx_graph = self.cgt.dgl_to_networkx(dgl_graph)
            list_of_augmented_edge_tensors.append(
                self.cgt.networkx_to_edge_tensor(nx_graph)
            )

        return list_of_augmented_edge_tensors
