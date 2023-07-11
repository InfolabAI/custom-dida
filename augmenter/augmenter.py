import torch
import utils_TiaRa
from convert_graph_types import ConvertGraphTypes
from tqdm import tqdm


class Augmenter:
    def __init__(self, data):
        """
        Parameters
        --------
        data: dict of keys ['x', 'e', 'train', 'val', 'test'] where x means node features, e means edge features, and train, val, test means train, validation, test split.
        """
        self.cgt = ConvertGraphTypes()
        self.list_of_dgl_graphs = []
        self.data = data
        num_nodes = self.data["x"].shape[0]
        original_edge_tensors = self.data["train"][
            "pedges"
        ]  # original_edge_tensors: [[2, num_edges], ...]
        original_edge_weights = self.data["train"][
            "weights"
        ]  # original_edge_tensors: [[2, num_edges], ...]

        # tqdm with description
        for edge_tensor, edge_weights in tqdm(
            zip(original_edge_tensors, original_edge_weights),
            desc="Converting the original edge tensors into DGL graphs...",
        ):
            nx_graph = self.cgt.edge_tensor_to_networkx(
                edge_tensor, edge_weights, num_nodes
            )
            self.list_of_dgl_graphs.append(self.cgt.networkx_to_dgl(nx_graph))

    def __call__(self):
        augmented_list_of_dgl_graphs = self._augment(self.list_of_dgl_graphs)
        # remove self loop
        augmented_list_of_dgl_graphs = [
            dgl_graph.remove_self_loop() for dgl_graph in augmented_list_of_dgl_graphs
        ]
        for t, (G, augG) in enumerate(
            zip(self.list_of_dgl_graphs, augmented_list_of_dgl_graphs)
        ):
            print(
                f"#edges at t={t}: {G.edges()[0].shape[0]} -aug-> {augG.edges()[0].shape[0]} : diff+ {augG.edges()[0].shape[0] - G.edges()[0].shape[0]}"
            )

        list_of_augmented_edge_tensors = []
        list_of_augmented_edge_weights = []
        for dgl_graph in tqdm(
            augmented_list_of_dgl_graphs,
            desc="Converting the augmented DGL graphs into edge tensors...",
        ):
            nx_graph = self.cgt.dgl_to_networkx(dgl_graph)
            (
                augmented_edge_tensor,
                augmented_edge_weights,
            ) = self.cgt.networkx_to_edge_tensor(nx_graph)
            list_of_augmented_edge_tensors.append(augmented_edge_tensor)
            list_of_augmented_edge_weights.append(augmented_edge_weights)

        self.data["train"]["pedges"] = list_of_augmented_edge_tensors
        self.data["train"]["edge_index_list"] = list_of_augmented_edge_tensors
        self.data["train"]["weights"] = list_of_augmented_edge_weights
        return self.data

    def filter_matrix(self, X, eps, normalize=True):
        assert eps < 1.0
        if self.dense:
            X[X < eps] = 0.0
        else:
            X = utils_TiaRa.sparse_filter(X, eps)
        if normalize:
            return self.normalize(X, ord="col")
        else:
            return X

    def normalize(self, A, ord="row", dense=None):
        if dense is None:
            dense = self.dense

        N = A.shape[0]
        A = A if ord == "row" else A.transpose(0, 1)

        norm = self.row_sum(A, dense=dense)
        norm[norm <= 0] = 1
        if ord == "sym":
            norm = norm**0.5

        if dense:
            inv_D = torch.diag(1 / norm)
        else:
            inv_D = utils_TiaRa.sparse_diag(1 / norm)

        if ord == "sym":
            nA = inv_D @ A @ inv_D
        else:
            nA = inv_D @ A
        return nA if ord == "row" else nA.transpose(0, 1)

    def row_sum(self, A, dense=None):
        if dense is None:
            dense = self.dense
        if dense:
            return A.sum(dim=1)
        else:
            return torch.sparse.sum(A, dim=1).to_dense()
