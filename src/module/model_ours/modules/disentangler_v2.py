import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from loguru import logger
from .scatter_and_gather import ScatterAndGather


class Disentangler(nn.Module):
    def __init__(self, args, embed_dim, comp_len, comp_dim):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.comp_len = comp_len
        self.comp_dim = comp_dim
        self.encode_layer_norm = nn.LayerNorm(embed_dim)
        self.encode_final_layer_norm = nn.LayerNorm(comp_len * comp_dim)
        self.decode_norm = nn.LayerNorm(comp_dim)
        self.scga = ScatterAndGather(args, embed_dim)
        self.node_comp_mlps = nn.AdaptiveAvgPool1d(comp_dim)
        self.node_decomp_mlps = nn.AdaptiveAvgPool1d(embed_dim)
        self.ortho_loss = torch.zeros(1).squeeze(0).float().to(args.device)

    def encode(
        self,
        x,
        padded_node_mask,
        indices_subnodes,
        node_num,
        padded_edge_mask,
        time_entirenodes_emdim,
    ):
        """
        Parameters
        ----------
        x: torch.Tensor
            [#timestamps, #tokens (node + edge), embed_dim]
        """
        x = self.encode_layer_norm(x)
        self.num_tokens = x.shape[1]
        # [#timestamps, #tokens (activated nodes + edges), embed_dim] -> [#activated_nodes, embed_dim]
        compressed_x_list = []
        nodes = x[padded_node_mask, :]
        time_entirenodes_emdim = self.scga._to_entire(
            nodes,
            node_num,
            indices_subnodes,
            self.args.graphs,
            time_entirenodes_emdim,
            is_mlp=False,
        )
        if indices_subnodes is None:
            indices_subnodes = self.args.batched_data["indices_subnodes"]

        len_ = self.comp_len
        if self.training:
            # activated 횟수가 많은 node 순서대로 정렬 (unique 면서 sort 된 순서대로니 counts 의 index 가 node 번호와 같고, counts 를 다시 sort 했으니, 많은 순서대로 정렬한 것)
            sorted_act_nodes = (
                torch.cat(indices_subnodes).unique(
                    return_counts=True
                    # 1: counts
                )[1]
                # 1: sort 한 뒤의 indices
            ).sort(descending=True)[1]
            baskets = [[] for _ in range(len_)]
            # 각 basket 의 node 들의 activated 횟수가 유사하도록 배분
            for i, node in enumerate(sorted_act_nodes):
                baskets[i % len_].append(int(node))
            baskets = [np.array(x) for x in baskets]
            max_len = np.array([len(x) for x in baskets]).max()
            stacked_indices1 = np.stack(
                [np.pad(x, (0, max_len - len(x))) for x in baskets]
            )
            self.stacked_indices = stacked_indices1

        pooled = self.node_comp_mlps(time_entirenodes_emdim)
        pooled = (
            pooled[:, self.stacked_indices, :].sum(2) / self.stacked_indices.shape[1]
        ).reshape(pooled.shape[0], 1, -1)
        compressed_x = self.encode_final_layer_norm(pooled)
        # self.ortho_loss = self.orthogonality_loss(*compressed_x_list)

        return compressed_x

    def decode(self, x, padded_node_mask, padded_edge_mask):
        """
        Parameters
        ----------
        x: torch.Tensor
            [#timestamps, #tokens (node + edge), embed_dim]
        """
        time_entirenodes_emdim = torch.zeros(
            x.shape[0], self.args.num_nodes, self.comp_dim, device=x.device
        )
        # 중복된 index 에 넣으면 덮어씌워지므로, 중복된 index 가 없도록 앞 30개 뒤 30개 나눠서 + 로 넣음
        x_reshape = x.reshape(x.shape[0], -1, 1, self.comp_dim)
        time_entirenodes_emdim[
            :, self.stacked_indices[: self.stacked_indices.shape[0] // 2], :
        ] += x_reshape[:, : self.stacked_indices.shape[0] // 2, :, :]
        time_entirenodes_emdim[
            :, self.stacked_indices[self.stacked_indices.shape[0] // 2 :], :
        ] += x_reshape[:, self.stacked_indices.shape[0] // 2 :, :, :]
        # reshape and indexing at once
        time_entirenodes_emdim = self.node_decomp_mlps(
            self.decode_norm(time_entirenodes_emdim)
        )

        time_tokens_emdim = torch.zeros(
            x.shape[0], self.num_tokens, self.embed_dim, device=x.device
        )
        time_tokens_emdim[padded_node_mask, :] = self.scga._from_entire(
            time_entirenodes_emdim, self.args.batched_data
        )

        return time_tokens_emdim

    def orthogonality_loss(self, *tensors):
        """
        Compute orthogonality loss between multiple tensors.

        Args:
            *tensors (torch.Tensor): Input tensors to compute orthogonality loss for.

        Returns:
            torch.Tensor: Orthogonality loss value.
        """
        assert all(
            tensor.shape == tensors[0].shape for tensor in tensors
        ), "Input tensors must have the same shape."

        tensors = [tensor.flatten() for tensor in tensors]
        # Normalize the tensors to have unit norm
        tensors_norm = [F.normalize(tensor, p=2, dim=-1) for tensor in tensors]

        # Compute the dot product between the normalized tensors
        dot_products = []
        for i in range(len(tensors_norm) - 1):
            for j in range(1, len(tensors_norm)):
                dot_products.append(
                    torch.sum(tensors_norm[i] * tensors_norm[j])
                    / torch.sum(tensors_norm[i] + tensors_norm[j])
                )

        # The orthogonality loss is the mean squared error between the dot products and zero
        ortho_loss = torch.mean(
            torch.stack([torch.pow(dot_product, 2) for dot_product in dot_products])
        )

        return ortho_loss
