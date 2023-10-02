# the difference from v6 is that v8 uses only one mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
        self.node_comp_mlps = nn.Sequential(
            nn.Linear(embed_dim, self.comp_dim * 2),
            nn.GELU(),
            nn.Dropout1d(0.1),
            nn.Linear(self.comp_dim * 2, self.comp_dim),
        )
        self.node_decomp_mlps = nn.Sequential(
            nn.Linear(self.comp_dim, self.comp_dim * 2),
            nn.GELU(),
            nn.Dropout1d(0.1),
            nn.Linear(self.comp_dim * 2, embed_dim),
        )
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

        entire_indices = np.arange(time_entirenodes_emdim.shape[1])
        # shuffle 하면 test AUC 0.6 이 한계, shuffle 안하면 0.7 가능.
        # if self.training:
        #    np.random.shuffle(entire_indices)
        self.indices_history = np.array_split(entire_indices, self.comp_len)
        for i, indices in enumerate(self.indices_history):
            compressed_x_list.append(
                self.node_comp_mlps(time_entirenodes_emdim[:, indices, :]).sum(
                    1, keepdim=True
                )
            )
        compressed_x = self.encode_final_layer_norm(torch.cat(compressed_x_list, dim=2))

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
        for indices, tensor in zip(
            self.indices_history, torch.split(x, self.comp_dim, dim=2)
        ):
            time_entirenodes_emdim[:, indices, :] += tensor
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
