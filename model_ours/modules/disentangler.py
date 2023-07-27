import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import zip_longest
from .scatter_and_gather import ScatterAndGather


class Disentangler(nn.Module):
    def __init__(self, args, embed_dim, comp_len, comp_dim):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.comp_len = comp_len
        self.comp_dim = comp_dim
        self.encode_layer_norm = nn.LayerNorm(embed_dim)
        self.encode_final_layer_norm = nn.LayerNorm(
            comp_len * comp_dim * (args.len_train - 1)
        )
        self.decode_norm = nn.LayerNorm(embed_dim)
        self.scga = ScatterAndGather(args, embed_dim)
        self.node_comp_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, self.comp_dim * 2),
                    nn.GELU(),
                    # nn.Dropout1d(0.1),
                    nn.Linear(self.comp_dim * 2, self.comp_dim),
                )
                for _ in range(self.comp_len)
            ]
        )
        self.node_decomp_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(comp_dim, comp_dim * 2),
                    nn.GELU(),
                    # nn.Dropout1d(0.1),
                    nn.Linear(comp_dim * 2, embed_dim),
                )
                for _ in range(self.comp_len)
            ]
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
        tee = time_entirenodes_emdim

        t_feat_list = []
        self.indices_history = []
        for t in range(tee.shape[0]):
            activated_node_indices = self.args.batched_data["indices_subnodes"][t]
            deactivated_node_indices = np.setdiff1d(
                np.arange(tee.shape[1]), activated_node_indices.numpy()
            )
            ac_feat = tee[:, activated_node_indices, :].sum(1, keepdim=True) / len(
                activated_node_indices
            )
            ac_feat = self.node_comp_mlps[0](ac_feat)
            deac_feat = tee[:, deactivated_node_indices, :].sum(1, keepdim=True) / len(
                deactivated_node_indices
            )
            deac_feat = self.node_comp_mlps[1](deac_feat)
            t_feat = torch.concat([ac_feat, deac_feat], dim=2)
            t_feat_list.append(t_feat)
            self.indices_history.append(
                (activated_node_indices, deactivated_node_indices)
            )

        feat_cat = torch.cat(t_feat_list[: self.args.len_train - 1], dim=2)
        feat = self.encode_final_layer_norm(feat_cat)
        self.ortho_loss = self.orthogonality_loss(*t_feat_list)

        return feat

    def decode(self, x, padded_node_mask, padded_edge_mask):
        """
        Parameters
        ----------
        x: torch.Tensor
            [#timestamps, #tokens (node + edge), embed_dim]
        """
        tee_list = []
        for t, (indices, t_x) in enumerate(
            zip_longest(self.indices_history, torch.split(x, self.comp_dim * 2, dim=2))
        ):
            t_tee = torch.zeros(1, self.args.num_nodes, self.embed_dim, device=x.device)
            if t_x is None:
                tee_list.insert(0, t_tee)
                continue
            [ac_feat, deac_feat] = torch.split(
                t_x[t].unsqueeze(0), self.comp_dim, dim=2
            )
            ac_feat = self.node_decomp_mlps[0](ac_feat)
            deac_feat = self.node_decomp_mlps[1](deac_feat)
            t_tee[:, indices[0], :] += ac_feat / len(indices[0])
            t_tee[:, indices[1], :] += deac_feat / len(indices[1])
            tee_list.append(t_tee)

        tee = self.decode_norm(torch.concat(tee_list, dim=0))

        return tee

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
