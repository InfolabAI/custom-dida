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
            nn.Dropout1d(0.5),
            nn.Linear(2 * embed_dim, comp_dim),
        )
        self.layer_norm_u = nn.LayerNorm(comp_dim)
        self.mlp_u = nn.Sequential(
            nn.Linear(comp_dim, 2 * comp_dim),
            nn.GELU(),
            nn.Linear(2 * comp_dim, embed_dim),
        )
        self.step = 0

    def _to_entire(self, x, batched_data, entire_features=None, is_mlp=True):
        offset = 0
        t_entire_embeddings = []
        ta_mean, bd_mean = 0, 0
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

            ta_mean += ta.abs().mean()
            bd_mean += bd.abs().mean()

            input_ = ta + bd
            t_embedding = self.mlp_d(self.layer_norm_d(input_)) if is_mlp else input_

            t_entire_embeddings.append(t_embedding)

        ret = torch.stack(t_entire_embeddings, dim=0)
        if entire_features is not None:
            ret = ret + entire_features
            logger.info(
                f"ret: {ret.abs().mean():.2f} entire_features: {entire_features.abs().mean():.2f} ta_mean: {ta_mean:.2f}, bd_mean: {bd_mean:.2f}"
            )
        else:
            pass
            # logger.info("entire_features is None")
        return ret

    def _from_entire(self, x, batched_data):
        node_features = []
        for nodes, activated_indices in zip(x, batched_data["indices_subnodes"]):
            node_features.append(nodes[activated_indices.long()])
        return self.mlp_u(self.layer_norm_u(torch.concat(node_features, dim=0)))
