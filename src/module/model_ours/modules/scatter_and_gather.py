import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from torch_scatter import scatter


class ScatterAndGather(nn.Module):
    def __init__(self, args, embed_dim, comp_dim=None):
        super().__init__()

        if comp_dim is None:
            comp_dim = embed_dim

        self.comp_dim = comp_dim
        self.args = args
        self.layer_norm_d = nn.LayerNorm(embed_dim)
        self.mlp_d = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.GELU(),
            # nn.Dropout1d(0.5),
            nn.Linear(2 * embed_dim, comp_dim),
        )
        self.layer_norm_u = nn.LayerNorm(comp_dim)
        self.mlp_u = nn.Sequential(
            nn.Linear(comp_dim, 2 * comp_dim),
            nn.GELU(),
            nn.Linear(2 * comp_dim, embed_dim),
        )
        self.step = 0

    def _to_entire(
        self,
        x_list,
        total_indices_subnodes,
        graphs,
    ):
        t_entire_embeddings = []
        for t, (x, activated_indices) in enumerate(
            zip(x_list, total_indices_subnodes)
        ):
            # t_activated_embedding.size == [#nodes at t, embed_dim]
            t_activated_embedding = scatter(
                # [#activated nodes at t, embed_dim]
                x,
                activated_indices.long().to(self.args.device),
                dim=0,
                dim_size=self.args.num_nodes,
                reduce="add",
            )

            ta = t_activated_embedding
            bd = graphs[t].ndata["X"]

            input_ = ta + bd
            t_embedding = self.mlp_d(self.layer_norm_d(input_))
            t_entire_embeddings.append(t_embedding)

        ret = torch.stack(t_entire_embeddings, dim=0)
        return ret

    def _from_entire(self, x, batched_data):
        node_features = []
        for nodes, activated_indices in zip(x, batched_data["indices_subnodes"]):
            node_features.append(nodes[activated_indices.long()])
        return self.mlp_u(self.layer_norm_u(torch.concat(node_features, dim=0)))
