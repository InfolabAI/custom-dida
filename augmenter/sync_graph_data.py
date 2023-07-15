import torch
import utils_TiaRa
from convert_graph_types import ConvertGraphTypes
from tqdm import tqdm


class SyncGraphData:
    def __init__(self, args, data):
        """
        Parameters
        --------
        data: dict of keys ['x', 'e', 'train', 'val', 'test'] where x means node features, e means edge features, and train, val, test means train, validation, test split.
        sync_type: str, one of ['todgl', 'todict']
        """
        if "dida" in args.model or "tokengt" in args.model:
            self.sync_type = "todict"
        elif "ours" in args.model:
            self.sync_type = "todgl"
        else:
            raise NotImplementedError(f"args.model: {args.model}")

        # convert dict to list of dgl graphs
        self.cgt = ConvertGraphTypes()
        self.data = data
        self.list_of_dgl_graphs = self.cgt.dict_to_list_of_dglG(data, args.device)

    def _sync(self, augmented_list_of_dgl_graphs):
        if self.sync_type == "todgl":
            return augmented_list_of_dgl_graphs
        elif self.sync_type == "todict":
            return self._dgl_graph_list_to_dict(augmented_list_of_dgl_graphs)
        else:
            raise NotImplementedError("sync_type must be one of ['todgl', 'todict']")

    def _dgl_graph_list_to_dict(self, augmented_list_of_dgl_graphs):
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
        return self.data["train"]

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
        return self._sync(augmented_list_of_dgl_graphs)

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
