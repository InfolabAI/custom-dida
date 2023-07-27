import torch
import torch.nn as nn
import math
import dgl

from torch_scatter import scatter
from loguru import logger
from time import time
from tqdm import tqdm
from convert_graph_types import ConvertGraphTypes
from model_ours.modules.scatter_and_gather import ScatterAndGather
from model_ours.modules.model_tokengt import TokenGTModel
from model_ours.modules.multihead_attention import MultiheadAttention
from model_ours.trainer_ours import TrainerOurs
from model_ours.tester_ours import TesterOurs
from utils_main import MultiplyPredictor


class CustomMultiheadAttention(MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        attention_dropout,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            attention_dropout=attention_dropout,
            self_attention=True,
        )

    def forward(self, query, value: list):
        """
        Parameters
        ----------
        query: [t, 1, embed_dim]
        value: list of dgl.graph
        """
        attn_probs = super().forward(
            query, query, query, ret_attn_probs=True, attn_bias=None
        )
        # mean of multi-heads [batch_size (1) * num_heads, T, T] -> [T, T]
        attn_probs = attn_probs.mean(0)
        # 평균보다 낮은 attention 은 mask(모델이 선택과 집중을 하도록)
        attn_probs[attn_probs < 1 / attn_probs.shape[0]] = 0
        # lower triangular matrix [T, T]
        mask = torch.tril(torch.ones_like(attn_probs).to(attn_probs.device))
        # 대각선은 1(t 자신의 adj 에 대해서는 attention 보장)
        attn_probs += torch.eye(attn_probs.shape[0], device=attn_probs.device)
        attn_probs = attn_probs * mask
        logger.debug(f"#nonzero attn_probs: {attn_probs.nonzero().shape[0]}")

        # get At_stack [T, #nodes, #nodes]
        At_stack = torch.stack(
            [
                torch.sparse_coo_tensor(
                    G.adj().indices(), G.adj().val, G.adj().shape, device=G.adj().device
                )
                for G in value
            ]
        )

        new_At_list = []
        for attn in attn_probs:
            # attn [T]
            new_At = self._propagate_edge(value, attn)
            new_At_list.append(new_At)

        return new_At_list

    def _propagate_edge(self, list_of_dgl_graphs, attn):
        """
        Parameters
        ----------
        list_of_dgl_graphs: list of dgl.graph
        """
        # reverse list_of_dgl_graphs
        new_At = None
        for graph, a_attn in zip(list_of_dgl_graphs, attn):
            if float(a_attn.cpu().detach()) == 0:
                continue
            At = graph.adj()
            At = torch.sparse_coo_tensor(
                At.indices(), At.val, At.shape, device=At.device
            )
            if new_At is None:
                new_At = At * a_attn
            else:
                new_At += At * a_attn
                # comm_At = comm_At.matmul(At * action)

        if new_At is None:
            breakpoint()

        return new_At


class Attention(nn.Module):
    def __init__(self, args, num_nodes):
        super().__init__()
        self.args = args
        self.mlp_in = nn.Sequential(
            nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 2),
            nn.ReLU(),
        )
        self.mlp_out = nn.Sequential(
            nn.Linear(args.encoder_embed_dim * 2, args.encoder_embed_dim),
            nn.ReLU(),
        )
        # self.bn1 = nn.BatchNorm1d(tokengt_args.encoder_embed_dim)
        # self.bn2 = nn.BatchNorm1d(tokengt_args.encoder_embed_dim)
        self.attention = CustomMultiheadAttention(
            args.encoder_embed_dim,
            args.encoder_attention_heads,
            attention_dropout=args.attention_dropout,
        )
        self.linear = nn.Linear(args.encoder_embed_dim, 1)
        self.load_positional_encoding(args.encoder_embed_dim, 1000, args.device)

    def forward(self, list_of_embeddings, list_of_dgl_graphs):
        """
        Parameters
        ----------
        list_of_embeddings: list of torch.tensor from self.main_model over entier timestamps [[#activated nodes, embed_dim], ...]
        """
        inputs = []
        for em in list_of_embeddings:
            # [#activated nodes, embed_dim] -> [#activated nodes, embed_dim*2]
            em = self.mlp_in(em)
            # [#activated nodes, embed_dim*2] -> [embed_dim*2, #activated nodes] -> [embed_dim*2, 1]
            em = torch.nn.functional.adaptive_avg_pool1d(
                em.transpose(0, 1), 1
            ) + torch.nn.functional.adaptive_max_pool1d(em.transpose(0, 1), 1)
            # [embed_dim*2, 1] -> [1, embed_dim*2] -> [1, embed_dim] -> [embed_dim]
            em = self.mlp_out(em.transpose(0, 1)).squeeze(0)
            inputs.append(em)

        # [t, embed_dim]
        inputs = torch.stack(inputs, dim=0)
        # positional encoding. self.pe.shape == [max_position, embed_dim]
        inputs = inputs + self.pe[: inputs.shape[0]]
        # [t, embed_dim] -> [t, 1, embed_dim]
        inputs = inputs.unsqueeze(1)
        # [t, 1, embed_dim] -> propagated_list_of_dgl_graphs
        return self.attention(query=inputs, value=list_of_dgl_graphs)

    def load_positional_encoding(self, dim_feature=1, max_position=1000, device="cpu"):
        """
        feature 의 위치를 추론에 포함하기 위해 positional embedding을 계산
        https://github.com/InfolabAI/References/blob/eef3666c88f9c4eb5117a0425652295eca012b0e/models/nezha/modeling_nezha.py#L154

        Args:
            d_model: feature의 dimension (현재 1)
            max_len: 위치의 최대값 (현재 window size)
        """
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_position, dim_feature).float()
        pe.require_grad = False

        position = torch.arange(0, max_position).float().unsqueeze(1)
        div_term = (
            torch.arange(0, dim_feature, 2).float() * -(math.log(10000.0) / dim_feature)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.to(device)


class OurModel(nn.Module):
    def __init__(self, args, data_to_prepare, num_nodes):
        super().__init__()
        self.main_model = TokenGTModel.build_model(args).to(args.device)
        args = self.main_model.args
        self.args = args
        self.attention = Attention(args, num_nodes)
        self.scatter_and_gather = ScatterAndGather(args, args.encoder_embed_dim)
        self.cs_decoder = MultiplyPredictor()

        self.trainer = TrainerOurs(args, self, data_to_prepare)
        self.tester = TesterOurs(args, self, data_to_prepare)
        self.cgt = ConvertGraphTypes()
        self.embeddings = None

    def _get_tr_input(self, list_of_dgl_graphs):
        """
        - list_of_dgl_graphs 를 TrInputDict (a set of subgraphs) 로 변환
            - subgraph_t == activated nodes at t
        """
        tr_input = self.cgt.dglG_list_to_TrInputDict(list_of_dgl_graphs)
        return tr_input

    def _get_propagated_graphs(self, list_of_dgl_graphs):
        """(DEPRECATED)
        한번에 모든 t에 대해 propagated graph 를 구하는 함수"""
        with torch.no_grad():
            tr_input = self._get_tr_input(list_of_dgl_graphs)
            # [sum(activated_nodes) of all the timestamps, embed_dim]
            embeddings = self.main_model(tr_input, get_embedding=True)

        list_of_embeddings = []
        offset = 0
        for node_num in tr_input["node_num"]:
            list_of_embeddings.append(embeddings[offset : offset + node_num])
            offset += node_num

        new_At_list = self.attention(list_of_embeddings, list_of_dgl_graphs)

        propagated_list_of_dgl_graphs = []
        for new_At, ori_dglG in zip(new_At_list, list_of_dgl_graphs):
            # attn [T] -> [T, 1, 1], then [T, 1, 1] * [T, #nodes, #nodes], then [T, #nodes, #nodes] -> [#nodes, #nodes]
            new_graph = self.cgt.weighted_adjacency_to_graph(
                new_At,
                # 사실 모든 ori_dglG 는 같은 ndata 를 가지고 있어서 [0]번째 nddata 를 게속 써도 상관없음
                ori_dglG.ndata["w"],
                # original edge num 의 1.5 배까지로 제한
                ori_dglG.adj().nnz * 1.5,
            )
            propagated_list_of_dgl_graphs.append(new_graph)

        return propagated_list_of_dgl_graphs

    def forward(self, list_of_dgl_graphs, epoch, is_train):
        tr_input = self._get_tr_input(list_of_dgl_graphs)
        self.args
        setattr(self.args, "batched_data", tr_input)
        setattr(self.args, "list_of_dgl_graphs", list_of_dgl_graphs)
        # [sum(activated_nodes) of all the timestamps, embed_dim]
        embeddings, time_entirenodes_emdim = self.main_model(
            tr_input, get_embedding=True
        )

        t_entire_embeddings = self.scatter_and_gather._to_entire(
            embeddings, tr_input, entire_features=time_entirenodes_emdim
        )
        return t_entire_embeddings, tr_input
