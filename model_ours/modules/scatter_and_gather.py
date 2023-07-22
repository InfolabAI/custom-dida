import torch
import torch.nn as nn
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

    def _to_entire(self, x, batched_data, entire_node_feature=None):
        offset = 0
        t_entire_embeddings = []
        comm_ta = 0
        comm_bd = 0
        comm_en = 0
        for t, (node_num, activated_indices) in enumerate(
            zip(batched_data["node_num"], batched_data["indices_subnodes"])
        ):
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
                input_ = ta + bd

            # t_embedding = self.mlp_d(self.layer_norm_d(input_))
            # t_embedding = self.mlp_d(input_)
            t_embedding = torch.nn.functional.adaptive_avg_pool1d(input_, self.comp_dim)
            t_entire_embeddings.append(t_embedding)
        # node_features are the same across all the timestamps, so, we use [0]
        # if entire_node_feature is not None:
        #     print(
        #         f"comm_ta/comm_bd: {comm_ta/(comm_bd + 1e-8)}, comm_en/comm_bd: {comm_en/(comm_bd + 1e-8)}"
        #     )
        return torch.stack(t_entire_embeddings, dim=0)

    def _from_entire(self, x, batched_data):
        node_features = []
        for entier_nodes, activated_indices in zip(x, batched_data["indices_subnodes"]):
            node_features.append(entier_nodes[activated_indices.long()])
        return self.mlp_u(self.layer_norm_u(torch.concat(node_features, dim=0)))
