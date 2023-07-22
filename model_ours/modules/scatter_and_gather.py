import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter


class ScatterAndGather(nn.Module):
    def __init__(self, args, embed_dim, comp_dim=None):
        super().__init__()

        if comp_dim is None:
            comp_dim = embed_dim

        self.comp_dim = comp_dim
        self.args = args
        self.layer_norm_d = nn.LayerNorm(embed_dim)
        self.layer_norm_ta = nn.LayerNorm(embed_dim)
        self.layer_norm_bd = nn.LayerNorm(embed_dim)
        self.layer_norm_en = nn.LayerNorm(1)
        self.mlp_d = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            nn.Dropout1d(0.5),
            nn.Linear(2 * embed_dim, comp_dim),
        )
        self.layer_norm_u = nn.LayerNorm(comp_dim)
        self.mlp_u = nn.Sequential(
            nn.Linear(comp_dim, 2 * comp_dim),
            nn.GELU(),
            nn.Linear(2 * comp_dim, embed_dim),
        )
        self.reduce_mlp = nn.Sequential(
            nn.Linear(comp_dim, 1),
            nn.GELU(),
        )
        self.step = 0

    def _to_entire(
        self, x, batched_data, entire_node_feature=None, use_bd=True, indices=None
    ):
        offset = 0
        t_entire_embeddings = []
        comm_ta = 0
        comm_bd = 0
        comm_en = 0
        comm_activated_nodes = []
        for t, (node_num, activated_indices) in enumerate(
            zip(batched_data["node_num"], batched_data["indices_subnodes"])
        ):
            comm_activated_nodes.append(activated_indices)
            # t_activated_embedding.size == [#nodes at t, embed_dim]
            t_activated_embedding = scatter(
                # [#activated nodes at t, embed_dim]
                x[offset : offset + node_num],
                activated_indices.long().to(self.args.device),
                dim=0,
                dim_size=self.args.num_nodes,
                reduce="add",
            )
            offset += node_num

            if indices is not None:
                ta = t_activated_embedding[indices]
                bd = batched_data["x"][indices]
            else:
                ta = t_activated_embedding
                bd = batched_data["x"]
            comm_ta += ta.abs().mean()
            comm_bd += bd.abs().mean()
            if entire_node_feature is not None:
                en = torch.nn.functional.sigmoid(
                    self.layer_norm_en(entire_node_feature[t])
                )
                comm_en += en.abs().mean()

                input_ = ta + bd * en
            else:
                if use_bd:
                    input_ = ta + bd
                else:
                    input_ = ta

            t_embedding = self.mlp_d(self.layer_norm_d(input_))
            # t_embedding = self.mlp_d(input_)
            # t_embedding = torch.nn.functional.adaptive_avg_pool1d(input_, self.comp_dim)
            t_entire_embeddings.append(t_embedding)
        # node_features are the same across all the timestamps, so, we use [0]
        # if entire_node_feature is not None:
        #     print(
        #         f"comm_ta/comm_bd: {comm_ta/(comm_bd + 1e-8)}, comm_en/comm_bd: {comm_en/(comm_bd + 1e-8)}"
        #     )
        return (
            torch.stack(t_entire_embeddings, dim=0),
            # 각 time 마다 activated 된 node 의 indices 를 모으면 중복된 node 가 있을 수 있으므로 unique
            torch.concat(comm_activated_nodes, dim=0).unique(),
        )

    def _from_entire(self, x, batched_data, top_activated_indices=None):
        node_features = []
        for nodes, activated_indices in zip(x, batched_data["indices_subnodes"]):
            if top_activated_indices is not None:
                # nodes 가 top activated nodes 라서 activated_indices 를 모두 포함하고 있지 않을 때
                ti = top_activated_indices.cpu().numpy()
                ai = activated_indices.cpu().numpy()
                intersection = np.intersect1d(ti, ai)
                # # nodes 에서의 intersection 위치
                ti_indices = np.where(np.isin(ti, intersection))[0]
                ai_indices = np.where(np.isin(ai, intersection))[0]
                ti_nodes = nodes
                ai_nodes = torch.zeros(
                    [len(activated_indices), self.comp_dim], device=x.device
                )
                ai_nodes[ai_indices, :] = ti_nodes[ti_indices, :]
                node_features.append(ai_nodes)
            else:
                # nodes 가 entire nodes 라서 activated_indices 를 모두 포함하고 있을 때
                node_features.append(nodes[activated_indices.long()])
        return self.mlp_u(self.layer_norm_u(torch.concat(node_features, dim=0)))
