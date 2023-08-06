import torch
import torch.nn as nn
import copy

from loguru import logger
from ..convert_graph_types import ConvertGraphTypes
from .modules.scatter_and_gather import ScatterAndGather
from .modules.model_tokengt import TokenGTModel
from ..utils_main import MultiplyPredictor

# from model_ours.modules.sample_nodes import SampleNodes
from .modules.sample_nodes_reduced import SampleNodes


class ORFApplier(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.orf_encoder = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

        # from orf.py
        q, r = torch.linalg.qr(
            torch.randn((10, args.num_nodes, hidden_dim * 2), device=args.device),
            mode="reduced",
        )
        # [1, num_entire_nodes, hidden_dim] -> [num_entire_nodes, hidden_dim]
        self.orf = torch.nn.functional.normalize(q, p=2, dim=2).to(args.device)

        self.step = 0

    def forward(self, x):
        x = x + self.orf_encoder(self.orf[self.step % 10])
        self.step += 1
        return x


class GeneratePool:
    def __init__(self, args):
        self.args = args

    def generate(self, list_of_dgl_graphs):
        new_At_list = []
        new_At = None
        for graph in list_of_dgl_graphs:
            At = graph.adj()
            At = torch.sparse_coo_tensor(
                At.indices(), At.val, At.shape, device=At.device
            )
            if new_At is None:
                new_At = At
            else:
                new_At = At + At.matmul(new_At)

            # logger.info(
            #     f"newAt #edges(newAt/At), newAt #nodes(newAt/At): {new_At.coalesce().values().shape[0]}({new_At.coalesce().values().shape[0]/At.coalesce().values().shape[0]:.2f}) , {new_At.coalesce().indices().unique().shape[0]}({new_At.coalesce().indices().unique().shape[0]/At.coalesce().indices().unique().shape[0]:.2f})"
            # )
            new_At_list.append(new_At)

        return new_At_list


class OurModel(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        args = _kwargs["args"]
        args.num_nodes = _kwargs["graphs"].num_nodes
        self.args = args
        self.cgt = ConvertGraphTypes()
        self.gp = GeneratePool(args)
        self.tr_input_pool = self.cgt.dglG_list_to_pool(
            self._get_graphs_for_pool(_kwargs["graphs"])
        )

        setattr(args, "batched_data_pool", self.tr_input_pool)

        self.main_model = TokenGTModel.build_model(args).to(args.device)
        self.sample_nodes = SampleNodes(args)
        self.orf_applier = ORFApplier(args, args.encoder_embed_dim)
        self.scatter_and_gather = ScatterAndGather(args, args.encoder_embed_dim)
        self.cs_decoder = MultiplyPredictor()

        self.tr_input_pool = None
        self.tr_input = None

    def _get_graphs_for_pool(self, list_of_dgl_graphs):
        """(DEPRECATED)
        한번에 모든 t에 대해 propagated graph 를 구하는 함수"""
        new_At_list = self.gp.generate(list_of_dgl_graphs)

        list_of_dgl_graphs_for_pool = []
        for new_At, ori_dglG in zip(new_At_list, list_of_dgl_graphs):
            # attn [T] -> [T, 1, 1], then [T, 1, 1] * [T, #nodes, #nodes], then [T, #nodes, #nodes] -> [#nodes, #nodes]
            new_graph = self.cgt.weighted_adjacency_to_graph(
                new_At,
                # 모든 ori_dglG 는 다른 ndata 를 가지고 있음
                ori_dglG.ndata["X"],
            )
            list_of_dgl_graphs_for_pool.append(new_graph)

        return list_of_dgl_graphs_for_pool

    def forward(self, dataset, start, end):
        # logger.debug(f"start: {start}, end: {end}")
        # 여기서 dataset을 변경하면 안됨. trainer 에서 start, end 계산과 의존성이 있기 때문.
        graphs = dataset.graphs[:end]
        sampled_original_indices = None
        graphs = copy.deepcopy(graphs)

        for i in range(len(graphs)):
            graphs[i].ndata["X"] = self.orf_applier(graphs[i].ndata["X"])

        if self.args.num_division_edgeprop > 0:
            graphs, sampled_original_indices = self.sample_nodes(graphs)

        self.tr_input = self.cgt.dglG_list_to_TrInputDict(
            graphs, sampled_original_indices
        )

        # setattr(
        #    self.args, "tokenizer", self.main_model.encoder.graph_encoder.graph_feature
        # )
        setattr(self.args, "batched_data", self.tr_input)
        setattr(self.args, "graphs", graphs)

        # [sum(activated_nodes) of all the timestamps, embed_dim]
        embeddings, list_ = self.main_model(self.tr_input, get_embedding=True)

        features = self.scatter_and_gather._to_entire(
            x=embeddings,
            total_node_num=list_[1],
            total_indices_subnodes=list_[2],
            graphs=graphs,
            entire_features=list_[0],
            is_mlp=True,
        )

        return features[start:, :, :]
