import torch
import torch.nn as nn
import math
import dgl
from loguru import logger
from time import time
from tqdm import tqdm
from convert_graph_types import ConvertGraphTypes
from community_dectection import CommunityDetection
from model_ours.modules.model_tokengt import TokenGTModel
from model_ours.modules.multihead_attention import MultiheadAttention
from model_ours.trainer_ours import TrainerOurs
from model_ours.tester_ours import TesterOurs


class MultiplyPredictor(torch.nn.Module):
    def __init__(self):
        super(MultiplyPredictor, self).__init__()

    def forward(self, z, e):
        x_i = z[e[0]]
        x_j = z[e[1]]
        x = (x_i * x_j).sum(dim=1)
        return torch.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, args, num_nodes):
        super().__init__()
        self.args = args
        self.stem = nn.Sequential(
            nn.Linear(num_nodes, 1),
            nn.ReLU(),
        )
        # self.bn1 = nn.BatchNorm1d(tokengt_args.encoder_embed_dim)
        # self.bn2 = nn.BatchNorm1d(tokengt_args.encoder_embed_dim)
        self.attention = MultiheadAttention(
            args.encoder_embed_dim,
            args.encoder_attention_heads,
            attention_dropout=args.attention_dropout,
            dropout=args.dropout,
            self_attention=True,
        )
        self.linear = nn.Linear(args.encoder_embed_dim, 1)
        self.load_positional_encoding(args.encoder_embed_dim, 1000, args.device)

    def forward(self, embeddings):
        """
        Parameters
        ----------
        embeddings: torch.tensor, shape [num_nodes, embed_dim]: 가장 마지막이 가장 최신 t 의 embedding 이라 가정함

        History
        -------
        - embedding 1개 [#nodes, embed_dim] -> [embed_dim, #nodes] -pooling-> [embed_dim] -모든 embedding stack-> [1, t, embed_dim] -self.attention-> [1, t, embed_dim] -squeeze-> [t, embed_dim] -linear-> [t, 1] -squeeze-> [t]
            - 이렇게 pooling 을 먼저 하면 정보의 손실이 커서 그런지 학습할수록 self.attention weight 는 0에 수렴하고, 모든 action 이 같은 값이 됨
        - embedding 1개 [#nodes, embed_dim] -모든 embedding stack, t()-> [t, embed_dim, #nodes] -linear, relu-> [t, embed_dim, 1] -t(), self.attention-> [1, t, embed_dim] -squeeze-> [t, embed_dim] -linear-> [t, 1] -squeeze-> [t]
            - pooing 으로 인한 정보손실을 줄이기 위해 linear 로 대체했지만, 동일한 문제 발생
        - 위와 동일한 상태에서 [t, embed_dim] 에 대해 batchnorm1d(embed_dim) 추가
            - 동일한 문제 발생
        - alpha 을 이용하는 연산을 adjacenty matrix 간 a_1*At_1 + a_2*At_2 가 아니라 (a_2*At_2).matmul(a_1*At_1) 로 변환
            - 동일한 문제 발생
        - positional encoding 추가
            - 동일한 문제 발생
        - alpha 에 sigmoid 가 아닌 softmax 사용
            - 동일한 문제 발생
        - collab 뿐 아니라 yelp 에 대해 실험 수행
            - 동일한 문제 발생
        """
        inputs = []
        # NOTE reversed 가 중요 action 은 t-1, t-2 순으로 나와야 함
        for i in reversed(range(len(embeddings))):
            inputs.append(embeddings[i])

        # [t, #nodes, embed_dim]
        inputs = torch.stack(inputs, dim=0)
        # [t, #nodes, embed_dim] -> [t, embed_dim, #nodes] -> [t, embed_dim, 1] -> [t, embed_dim]
        inputs = self.stem(inputs.transpose(1, 2)).squeeze(2)
        # positional encoding. self.pe.shape == [max_position, embed_dim]
        inputs = inputs + self.pe[: inputs.shape[0]]
        # [t, embed_dim] -> [1, t, embed_dim]
        inputs = inputs.unsqueeze(0)
        # attn [1, t, embed_dim], attn_weights [t, 1, 1]
        attn, attn_weights = self.attention(
            query=inputs, key=inputs, value=inputs, attn_bias=None
        )
        # [1, t, embed_dim] -> [t, embed_dim]
        attn = attn.squeeze(0)
        # [t, embed_dim] -> [t, 1] -> [1, t]
        attn = self.linear(attn).transpose(0, 1)
        if attn.shape[1] > 1 and self.args.alpha_std == 1:
            attn = attn / (attn.std() + 1e-8)
        # [1, t] -> [t]
        out = torch.nn.functional.softmax(attn, dim=1).squeeze(0)

        # reversed index 에 의한 계산이므로, out[0]==action(t-1), out[1]==action(t-2), ...
        return out

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
        self.cs_decoder = MultiplyPredictor()

        self.args = args
        self.trainer = TrainerOurs(args, self, data_to_prepare)
        self.tester = TesterOurs(args, self, data_to_prepare)
        self.cgt = ConvertGraphTypes()
        self.embeddings = None
        self.partition_dict = None
        self.total_step = 0  # for writer
        self.comm_groups = None

    def _get_augmented_graph(self, list_of_dgl_graphs, t):
        """
        한번에 하나의 t에 대해 new graph 를 구하는 함수"""
        if t == 0:
            return list_of_dgl_graphs[0]

        assert self.embeddings is not None, "get_graph_embeddings 를 먼저 실행해야 함"

        # actions for t=1 (action for t=0), t=2 (actions for t=1, 0), t=3 (actions for t=2, 1, 0), ...
        action = self._get_action(self.embeddings, t)
        self.trainer.runnerProperty.writer.add_histogram(
            "action", action, self.total_step
        )
        if self.total_step == 0 or self.total_step % 100 == 0:
            self.trainer.runnerProperty.writer.add_text(
                "action", str(action), self.total_step
            )
        # NOTE [::-1] reversed indices to sync this to action
        sublist_of_dgl_graphs = list_of_dgl_graphs[:t][::-1]
        cur_graph = list_of_dgl_graphs[t]
        augmented_graph = self._get_propagated_graph(
            sublist_of_dgl_graphs, action, cur_graph
        )

        return augmented_graph

    def _get_graph_embeddings(self, list_of_dgl_graphs, comm_group_id):
        """
        모든 t 의 original graph 에 대해 embedding 을 구하는 함수
        한번에 구하는 것이 필요한 이유는 t 에 대한 action 을 구하기 위해, t-1, t-2, ... 의 embedding 이 필요하기 때문
        """
        embeddings = []
        with torch.no_grad():
            for t, dglG in tqdm(
                enumerate(list_of_dgl_graphs),
                desc="get_graph_embeddings",
                total=len(list_of_dgl_graphs),
                leave=False,
            ):
                st = time()
                # NOTE original graphs 에 대해 partition 을 한 번 저장하고 나면, 다음부터는 저장된 partition 을 사용
                a_graph_at_t = self._get_tr_input(dglG, comm_group_id)
                self.trainer.runnerProperty.writer.add_scalar(
                    "get_tr_input time", time() - st, self.total_step
                )
                st = time()
                embeddings.append(self.main_model(a_graph_at_t, get_embedding=True))
                self.trainer.runnerProperty.writer.add_scalar(
                    "main_model time", time() - st, self.total_step
                )

        self.embeddings = embeddings

    def _get_tr_input(self, dglG, comm_group_id):
        """
        - dglG 를 TrInputDict 로 변환하는 함수

        Parameters
        ----------
        dglG: dgl graph
        comm_group_id: int: 사전에 생성된 comm_groups 에서 몇 번째 comm_group 을 subgraph sampling 에 사용할 것인지에 대한 index
        """
        dglG = dglG.to(self.args.device)
        if self.args.dont_use_subgraph:
            graph_dict = self._get_graph_dict(dglG)
        else:
            graph_dict = self._get_subgraphs_dict_and_comm_groups(dglG, comm_group_id)
        return graph_dict

    def _get_graph_dict(self, dglG):
        a_graph_at_t = self.cgt.dglG_to_TrInputDict_NoSubgraphs(dglG)
        return a_graph_at_t

    def _get_subgraphs_dict_and_comm_groups(self, dglG, comm_group_id):
        assert self.comm_groups is not None, "_generate_community_groups() 를 먼저 실행해야 함"

        st = time()
        a_graph_at_t = self.cgt.dglG_to_TrInputDict(
            dglG, self.comm_groups[comm_group_id]
        )
        self.trainer.runnerProperty.writer.add_scalar(
            "dgltotrinput time", time() - st, self.total_step
        )
        return a_graph_at_t

    def _get_actions(self, list_of_dgl_graphs):
        """(DEPRECATED)
        한번에 모든 t에 대해 action 을 구하는 함수"""
        embeddings = self.get_graph_embeddings(list_of_dgl_graphs)
        action_list = []
        for t in range(1, len(embeddings) - 1):
            action_list.append(self.attention(embeddings[:t]))
        return action_list

    def _get_action(self, entier_embeddings, t):
        """
        한번에 하나의 t에 대해 {t-1, t-2, ...} action 을 구하는 함수
        Parameters
        ----------
        entier_embeddings: list of torch.tensor from self.main_model over entier timestamps
        """
        return self.attention(entier_embeddings[:t])

    def _get_propagated_graph(
        self, sublist_of_dgl_graphs, actions, cur_graph, edge_number_limit_ratio=1.5
    ):
        """
        Parameters
        ----------
        sublist_of_dgl_graphs: list of dgl.graph
        actions: list of int: For t, actions means action for t-1, t-2, ..., 0
        cur_graph: dgl.graph
        edge_number_limit_ratio: float: If this is set to 1.5, the number of augmented edges are limited to 1.5 times of the number of original edges
        """
        # reverse list_of_dgl_graphs
        comm_At = None
        for graph, action in zip(sublist_of_dgl_graphs, actions):
            At = graph.adj()
            At = torch.sparse_coo_tensor(
                At.indices(), At.val, At.shape, device=At.device
            )
            if comm_At is None:
                comm_At = At * action
            else:
                comm_At += At * action
                # comm_At = comm_At.matmul(At * action)

        cur_adj = cur_graph.adj()
        cur_At = torch.sparse_coo_tensor(
            cur_adj.indices(), cur_adj.val, cur_adj.shape, device=cur_adj.device
        )
        cur_At = cur_At + comm_At
        new_graph = self.cgt.weighted_adjacency_to_graph(
            cur_At.transpose(1, 0).coalesce(),
            cur_graph.ndata["w"],
            cur_adj.nnz * edge_number_limit_ratio,
        )

        self.trainer.runnerProperty.writer.add_scalar(
            "Ratio of #orig_edges to $aug_edges",
            new_graph.adj().nnz / cur_adj.nnz,
            self.total_step,
        )
        return new_graph

    def get_augmented_graphs(self, list_of_dgl_graphs):
        """(DEPRECATED)
        한번에 모든 t에 대해 augmented graph 를 구하는 함수"""
        # deprecated
        # actions for t=1 (action for t=0), t=2 (actions for t=1, 0), t=3 (actions for t=2, 1, 0), ...
        action_list = self._get_actions(list_of_dgl_graphs)
        t = 1
        augmented_graphs = [list_of_dgl_graphs[0]]

        for actions in action_list:
            # [::-1] reversed indices
            sublist_of_dgl_graphs = list_of_dgl_graphs[:t][::-1]
            cur_graph = list_of_dgl_graphs[t]
            new_graph = self._get_propagated_graph(
                sublist_of_dgl_graphs, actions, cur_graph
            )
            augmented_graphs.append(new_graph)

            t += 1

        return augmented_graphs

    def _generate_community_groups(self, list_of_dgl_graphs, num_comm_groups):
        """
        모든 dglG 의 edge 가 병합된 dglG 에 대해 num_comm_groups 만큼 community group 을 생성하는 함수
        """
        union_graph = dgl.merge(list_of_dgl_graphs)
        logger.info(f"union_graph edges: {union_graph.edges()[0].shape[0]}")
        union_graph = union_graph.remove_self_loop()
        # removing parallel edges like [i,j] and [j,i]
        # union_graph = dgl.to_simple(union_graph)
        cd = CommunityDetection(self.args, union_graph)
        comm_groups = cd.get_merged_communities(num_comm_groups=num_comm_groups)
        self.comm_groups = comm_groups

    def forward(self, list_of_dgl_graphs, t, epoch, is_train):
        self.total_step += 1
        comm_group_id = epoch % self.args.num_comm_groups
        if epoch == 1 and t == 0 and is_train:
            self._generate_community_groups(
                list_of_dgl_graphs, self.args.num_comm_groups
            )

        if epoch == 1 or self.args.propagate != "dyaug":
            dglG = list_of_dgl_graphs[t]
        else:
            if t == 0 and is_train:
                self._get_graph_embeddings(list_of_dgl_graphs, comm_group_id)
            dglG = self._get_augmented_graph(list_of_dgl_graphs, t)

        graph = self._get_tr_input(dglG, comm_group_id)

        st = time()
        embedding = self.main_model(graph)
        logger.debug(f"main_model time: {time() - st}")
        return embedding
