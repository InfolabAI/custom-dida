import torch
import dgl
from torch.nn import Parameter
from .modules.scatter_and_gather import ScatterAndGather
from .modules.tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer
from torch_geometric.nn.inits import glorot, zeros
from utils import normalize_graph
from loguru import logger
from ..convert_graph_types import ConvertGraphTypes
from .modules.tokenizer import GraphFeatureTokenizer


def get_layer():
    return TokenGTGraphEncoderLayer()


class TRRN(torch.nn.Module):
    def __init__(
        self,
        input_dim=32,
        hidden_dim=32,
        output_dim=32,
        rnn="LSTM",
        num_layers=3,
        dropout=0,
        dropedge=0,
        renorm_order="sym",
        device="cuda",
        **_kwargs,
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
        rnn
            RNN model
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

        dimensions = [input_dim] + \
            (num_layers - 1) * [hidden_dim] + [output_dim]
        self.layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout)
        self.dropedge = dgl.DropEdge(dropedge) if dropedge > 0 else None
        self.renorm_order = renorm_order
        self.rnn = rnn
        self.act = torch.nn.ReLU()
        self.cgt = ConvertGraphTypes()
        self._sync_graphs(_kwargs["graphs"])
        self.args = _kwargs["args"]
        self.graph_feature = GraphFeatureTokenizer(
            args=self.args,
            hidden_dim=input_dim,
            n_layers=0,
        )
        self.scatter_and_gather = ScatterAndGather(
            self.args, input_dim)

        if rnn == "LSTM":
            GNN = GTRLSTM
        elif rnn == "GRU":
            GNN = GConvGRU
        else:
            raise NotImplementedError("no such RNN model {}".format(rnn))

        for layer in range(len(dimensions) - 1):
            self.layers.append(
                GNN(dimensions[layer], dimensions[layer + 1], normalize=False).to(
                    device
                )
            )
            self.norms.append(
                torch.nn.BatchNorm1d(dimensions[layer + 1], device=device)
            )

    def _sync_graphs(self, dataset):
        """
        - graphs 의 ndata 를 input_graphs 로 옮김
        - self_loop 을 제거
        - edata 를 [#edges, 1] -> [#edges, #embed_dim] 로 변경
        - graphs 를 input_graphs 로 변경"""
        for i in range(len(dataset)):
            dataset.input_graphs[i].ndata["X"] = dataset.graphs[i].ndata["X"]
            dataset.input_graphs[i] = dataset.input_graphs[i].remove_self_loop(
            )
            dataset.input_graphs[i].edata["w"] = (
                dataset.input_graphs[i]
                .edata["w"]
                .reshape(-1, 1)
                .broadcast_to(-1, dataset.input_graphs[0].ndata["X"].shape[1])
            )
            dataset.graphs[i] = dataset.input_graphs[i]

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

    def forward(self, dataset, start, end):
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
        input_graphs = dataset.input_graphs[:end]
        # 기본적으로 self.dropedge=None
        if self.training and self.dropedge:
            input_graphs = [
                normalize_graph(
                    self.dropedge(graph.remove_self_loop()).add_self_loop(),
                    self.renorm_order,
                )
                for graph in input_graphs
            ]

        subgraph_indices_list = []
        tokenized_input_list = []
        for t, tr_input in enumerate(self.cgt.dglG_list_to_TrInputDict_per_t(input_graphs)):
            subgraph_indices_list.append(tr_input["indices_subnodes"])
            tokenized_input_list.append(self.graph_feature(tr_input))

        Hs_tmp = []
        for layer, norm in zip(self.layers, self.norms):
            H = None
            C = None
            Hs = [tokenized_input[0]
                  for tokenized_input in tokenized_input_list] if len(Hs_tmp) == 0 else Hs_tmp

            Hs_tmp = []
            for t, H in enumerate(Hs):
                x = H
                if self.rnn == "LSTM":
                    H, C = layer(x, H=H, C=C)
                elif self.rnn == "GRU":
                    H = layer(x, H=H)
                else:
                    raise NotImplementedError(
                        "no such RNN model {}".format(self.rnn))

                Hs_tmp.append(H[:, 1:, :])

        lastHs = []
        for H, tokenized_input in zip(Hs_tmp, tokenized_input_list):
            lastHs.append(H[tokenized_input[3], :])

        # feature = self.dropout(self.act(self.norm(torch.stack(Hs), norm)))

        feature = self.scatter_and_gather._to_entire(
            x_list=lastHs,
            total_indices_subnodes=subgraph_indices_list,
            graphs=input_graphs,
        )

        """
        (Pdb) p feature.shape
        torch.Size([69, 7125, 32])
        """

        return feature[start:, :, :]


class GTRLSTM(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalize: bool = True,
        bias: bool = True,
    ):
        super(GTRLSTM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.bias = bias
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):
        self.tr_x_i = get_layer()
        self.tr_h_i = get_layer()

        self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):
        self.tr_x_f = get_layer()
        self.tr_h_f = get_layer()

        self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):
        self.tr_x_c = get_layer()
        self.tr_h_c = get_layer()

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):
        self.tr_x_o = get_layer()
        self.tr_h_o = get_layer()

        self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_c_i)
        glorot(self.w_c_f)
        glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_state(self, X, H_or_C):
        if H_or_C is None:
            Htoken = torch.zeros(X.shape[0], 1, X.shape[2]).to(X.device)
        else:
            Htoken = self._retrieve_HC_from_combined_X(H_or_C)
        return self._apply_H_or_C_to_X(X, Htoken)

    def _apply_H_or_C_to_X(self, X, H_or_C):
        return torch.concat([H_or_C, X], dim=1)

    def _retrieve_HC_from_combined_X(self, combined_X):
        H_or_C = combined_X[:, 0, :].unsqueeze(1)
        return H_or_C

    def _calculate_input_gate(self, X, H, C):
        I = self.tr_x_i(X)
        I = I + self.tr_h_i(H)
        I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(
        self,
        X,
        H,
        C,
    ):
        F = self.tr_x_f(X)
        F = F + self.tr_h_f(H)
        F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, H, C, I, F):
        T = self.tr_x_c(X)
        T = T + self.tr_h_c(H)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(
        self,
        X,
        H,
        C,
    ):
        O = self.tr_x_o(X)
        O = O + self.tr_h_o(H)
        O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state and cell state
        matrices are not present when the forward pass is called these are
        initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor, optional)* - Cell state matrix for all nodes.
            * **lambda_max** *(PyTorch Tensor, optional but mandatory if normalization is not sym)* - Largest eigenvalue of Laplacian.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
            * **C** *(PyTorch Float Tensor)* - Cell state matrix for all nodes.
        """
        """
        (Pdb) p X.shape
        torch.Size([7125, 32])
        (Pdb) p H.shape
        torch.Size([7125, 32])
        (Pdb) p C.shape
        torch.Size([7125, 32])
        (Pdb) p I.shape
        torch.Size([7125, 32])
        (Pdb) p F.shape
        torch.Size([7125, 32])
        (Pdb) p O.shape
        torch.Size([7125, 32])
        (Pdb) p H.shape
        torch.Size([7125, 32])
        """
        H = self._set_state(X, H)
        C = self._set_state(X, C)
        X = self._set_state(X, None)
        I = self._calculate_input_gate(X, H, C)
        F = self._calculate_forget_gate(X, H, C)
        C = self._calculate_cell_state(X, H, C, I, F)
        O = self._calculate_output_gate(X, H, C)
        H = self._calculate_hidden_state(O, C)
        return H, C
