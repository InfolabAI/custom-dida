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
        self.conv = nn.Sequential(
            nn.Conv2d(4, 4, 8, 8),
            nn.GELU(),
            # nn.Dropout(0.5),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 8, 8),
            nn.GELU(),
            # nn.Dropout(0.5),
        )
        self.node_comp_mlps = nn.Sequential(
            nn.Linear(args.num_nodes, 4000),
            nn.GELU(),
            # nn.Dropout(0.5),
        )
        self.node_decomp_mlps = nn.Sequential(
            nn.Linear(4000, args.num_nodes),
            nn.GELU(),
            # nn.Dropout(0.5),
        )
        self.ortho_loss = torch.zeros(1).squeeze(0).float().to(args.device)

    def encode(self, x, padded_node_mask, padded_edge_mask, time_entirenodes_emdim):
        """
        Parameters
        ----------
        x: torch.Tensor
            [#timestamps, #tokens (node + edge), embed_dim]
        """
        x = self.encode_layer_norm(x)
        # [#timestamps, #tokens (activated nodes + edges), embed_dim] -> [#activated_nodes, embed_dim]
        compressed_x_list = []
        nodes = x[padded_node_mask, :]
        time_entirenodes_emdim = self.scga._to_entire(
            nodes, self.args.batched_data, time_entirenodes_emdim, is_mlp=False
        )
        time_entirenodes_emdim = time_entirenodes_emdim[
            :, torch.randperm(time_entirenodes_emdim.size(1)), :
        ]
        time_entirenodes_emdim = time_entirenodes_emdim.transpose(1, 2).reshape(
            time_entirenodes_emdim.shape[0], 4, 8, -1
        )
        time_entirenodes_emdim = self.node_comp_mlps(time_entirenodes_emdim)
        compressed_x = (
            self.conv(time_entirenodes_emdim)
            .transpose(1, 2)
            .reshape(time_entirenodes_emdim.shape[0], 1, -1)
        )

        return compressed_x

    def decode(self, x, padded_node_mask, padded_edge_mask):
        """
        Parameters
        ----------
        x: torch.Tensor
            [#timestamps, #tokens (node + edge), embed_dim]
        """
        x = x.reshape(x.shape[0], 1, 4, 500).transpose(1, 2)
        x = self.deconv(x)
        x = self.node_decomp_mlps(x)
        x = x.reshape(x.shape[0], 32, -1).transpose(1, 2)
        return x

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
