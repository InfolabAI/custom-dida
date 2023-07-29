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

            logger.info(
                f"newAt #edges(newAt/At), newAt #nodes(newAt/At): {new_At.coalesce().values().shape[0]}({new_At.coalesce().values().shape[0]/At.coalesce().values().shape[0]:.2f}) , {new_At.coalesce().indices().unique().shape[0]}({new_At.coalesce().indices().unique().shape[0]/At.coalesce().indices().unique().shape[0]:.2f})"
            )
            new_At_list.append(new_At)

        return new_At_list


class OurModel(nn.Module):
    def __init__(self, args, data_to_prepare, list_of_dgl_graphs):
        super().__init__()
        self.cgt = ConvertGraphTypes()
        self.gp = GeneratePool(args)
        self.tr_input_pool = self.cgt.dglG_list_to_pool(
            self._get_graphs_for_pool(list_of_dgl_graphs)
        )

        setattr(args, "batched_data_pool", self.tr_input_pool)

        self.main_model = TokenGTModel.build_model(args).to(args.device)
        args = self.main_model.args
        self.args = args
        self.scatter_and_gather = ScatterAndGather(args, args.encoder_embed_dim)
        self.cs_decoder = MultiplyPredictor()

        self.trainer = TrainerOurs(args, self, data_to_prepare)
        self.tester = TesterOurs(args, self, data_to_prepare)
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
                # 사실 모든 ori_dglG 는 같은 ndata 를 가지고 있어서 [0]번째 nddata 를 게속 써도 상관없음
                ori_dglG.ndata["w"],
            )
            list_of_dgl_graphs_for_pool.append(new_graph)

        return list_of_dgl_graphs_for_pool

    def forward(self, list_of_dgl_graphs, epoch, is_train):
        self.tr_input = self.cgt.dglG_list_to_TrInputDict(list_of_dgl_graphs)

        setattr(
            self.args, "tokenizer", self.main_model.encoder.graph_encoder.graph_feature
        )
        setattr(self.args, "batched_data", self.tr_input)
        setattr(self.args, "list_of_dgl_graphs", list_of_dgl_graphs)
        # [sum(activated_nodes) of all the timestamps, embed_dim]
        embeddings, list_ = self.main_model(self.tr_input, get_embedding=True)

        if "att_x" in self.args.handling_time_att:
            tee = self.scatter_and_gather._to_entire(
                x=embeddings,
                total_node_num=list_[1],
                total_indices_subnodes=list_[2],
                original_x=self.args.batched_data["x"],
                entire_features=list_[0],
                is_mlp=True,
            )
        else:
            tee = list_[0]
        logger.debug("att_x")
        return tee, self.tr_input
