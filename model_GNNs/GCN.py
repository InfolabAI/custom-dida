import torch, dgl
from utils_main import normalize_graph, MultiplyPredictor
from .trainer_gnns import TrainerGNNs
from .tester_gnns import TesterGNNs
from loguru import logger


class GCN(torch.nn.Module):
    def __init__(
        self,
        args,
        data_to_prepare,
        input_dim=32,
        hidden_dim=32,
        output_dim=32,
        num_layers=3,
        dropout=0,
        dropedge=0,
        renorm_order="sym",
        device="cuda",
        **_kwargs
    ):
        """
        Parameters
        ----------
        input_dim
            input feature dimension
        hidden_dim
            hidden feature dimension
        output_dim
            output feature dimension
        num_layers
            number of layers
        dropout
            dropout ratio
        dropedge
            dropedge ratio
        renorm_order
            normalization order after dropedge
        device
            device name
        """
        super().__init__()

        dimensions = [input_dim] + (num_layers - 1) * [hidden_dim] + [output_dim]
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout)
        self.dropedge = dgl.DropEdge(dropedge) if dropedge > 0 else None
        self.renorm_order = renorm_order

        for layer in range(len(dimensions) - 1):
            self.layers.append(
                dgl.nn.GraphConv(
                    dimensions[layer], dimensions[layer + 1], activation=torch.nn.ReLU()
                ).to(device)
            )
            self.norms.append(
                torch.nn.BatchNorm1d(dimensions[layer + 1], device=device)
            )

        self.args = args
        self.cs_decoder = MultiplyPredictor()
        self.trainer = TrainerGNNs(args, self, data_to_prepare)
        self.tester = TesterGNNs(args, self, data_to_prepare)

    def norm(self, X, normfn):
        """
        Parameters
        ----------
        X
            (T, N, F) shape tensor
        normfn
            BatchNorm1d function

        Returns
        -------
        Normalized X
        """
        return normfn(X.permute(1, 2, 0)).permute(2, 0, 1)

    def forward(self, input_graphs, start, end):
        """
        Parameters
        ----------
        dataset
            temporal graph dataset
        start
            start time step of temporal dataset
        end
            end time step of temporal dataset

        Returns
        -------
        Embedding tensor
        """
        for i in range(len(input_graphs)):
            input_graphs[i] = dgl.add_self_loop(input_graphs[i])

        feature = torch.stack([graph.ndata["w"] for graph in input_graphs])

        for layer, norm in zip(self.layers, self.norms):
            feature = [
                layer(graph, feature[t, :, :], edge_weight=graph.edata["w"])
                for t, graph in enumerate(input_graphs)
            ]
            feature = self.dropout(self.norm(torch.stack(feature), norm))

        return feature[start:, :, :]
