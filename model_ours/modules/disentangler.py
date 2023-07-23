import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Disentangler(nn.Module):
    def __init__(self, args, embed_dim):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.comp_len = 16
        self.comp_dim = embed_dim // 2
        self.node_comp_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, self.comp_dim * 2),
                    nn.GELU(),
                    nn.Dropout1d(0.1),
                    nn.Linear(self.comp_dim * 2, self.comp_dim),
                )
                for _ in range(self.comp_len)
            ]
        )
        self.edge_comp_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, self.comp_dim * 2),
                    nn.GELU(),
                    nn.Dropout1d(0.1),
                    nn.Linear(self.comp_dim * 2, self.comp_dim),
                )
                for _ in range(self.comp_len)
            ]
        )
        self.ortho_loss = torch.zeros(1).squeeze(0).float().to(args.device)

    def forward(self, x, padded_node_mask, padded_edge_mask):
        """
        Parameters
        ----------
        x: torch.Tensor
            [#timestamps, #tokens (node + edge), embed_dim]
        """
        nodes = x[padded_node_mask, :]
        edges = x[padded_edge_mask, :]

        compressed_x_list = []

        # node 또느 edge 를 compress 하기 전에
        for mlp in self.node_comp_mlps:
            # node 정보 처리
            zeros_x = torch.zeros(
                x.shape[0], x.shape[1], self.comp_dim, device=x.device
            )
            zeros_nodes = torch.zeros(nodes.shape[0], self.comp_dim, device=x.device)
            rand_indices = np.random.choice(np.arange(nodes.shape[0]), size=1000)
            zeros_nodes[rand_indices, :] = mlp(nodes[rand_indices, :])
            zeros_x[padded_node_mask, :] = zeros_nodes
            zeros_x = torch.nn.functional.adaptive_avg_pool1d(
                zeros_x.transpose(1, 2), 1
            ).transpose(1, 2)
            compressed_x_list.append(zeros_x)

        for mlp in self.edge_comp_mlps:
            # edge 정보 처리
            zeros_x = torch.zeros(
                x.shape[0], x.shape[1], self.comp_dim, device=x.device
            )
            zeros_edges = torch.zeros(edges.shape[0], self.comp_dim, device=x.device)
            rand_indices = np.random.choice(np.arange(edges.shape[0]), size=1000)
            zeros_edges[rand_indices, :] = mlp(edges[rand_indices, :])
            zeros_x[padded_edge_mask, :] = zeros_edges
            zeros_x = torch.nn.functional.adaptive_avg_pool1d(
                zeros_x.transpose(1, 2), 1
            ).transpose(1, 2)
            compressed_x_list.append(zeros_x)
        compressed_x = torch.cat(compressed_x_list, dim=2)

        self.ortho_loss = self.orthogonality_loss(*compressed_x_list)

        return compressed_x

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

        # Normalize the tensors to have unit norm
        tensors_norm = [F.normalize(tensor, p=2, dim=-1) for tensor in tensors]

        # Compute the dot product between the normalized tensors
        dot_products = []
        for i in range(len(tensors_norm) - 1):
            for j in range(1, len(tensors_norm)):
                dot_products.append(
                    torch.sum(tensors_norm[i] * tensors_norm[j], dim=-1)
                )

        # The orthogonality loss is the mean squared error between the dot products and zero
        ortho_loss = torch.mean(
            torch.stack([torch.pow(dot_product, 2) for dot_product in dot_products])
        )

        return ortho_loss
