import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Disentangler(nn.Module):
    def __init__(self, args, embed_dim, comp_len, comp_dim):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.comp_len = comp_len
        self.comp_dim = comp_dim
        self.encode_layer_norm = nn.LayerNorm(embed_dim)
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
        self.node_decomp_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.comp_dim, self.comp_dim * 2),
                    nn.GELU(),
                    nn.Dropout1d(0.1),
                    nn.Linear(self.comp_dim * 2, embed_dim),
                )
                for _ in range(self.comp_len)
            ]
        )
        self.edge_decomp_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.comp_dim, self.comp_dim * 2),
                    nn.GELU(),
                    nn.Dropout1d(0.1),
                    nn.Linear(self.comp_dim * 2, embed_dim),
                )
                for _ in range(self.comp_len)
            ]
        )
        self.ortho_loss = torch.zeros(1).squeeze(0).float().to(args.device)

    def get_indices_from_baskets(self, padded_mask):
        # create baskets of node indices, each basket means the indices of nodes at a timestamp
        offset = 0
        baskets = []
        for mask in padded_mask:
            num_nodes_at_t = int(mask.sum().cpu())
            basket = np.arange(offset, offset + num_nodes_at_t)
            baskets.append(basket)
            offset += num_nodes_at_t

        indices_history = []
        len_baskets = [len(x) for x in baskets]
        for i in range(self.comp_len):
            # sample (1/self.comp_len) of nodes at each timestamp, and remove the sampled nodes from each basket
            sampled_nodes = []
            for i in range(len(baskets)):
                num_sampled_nodes = int(len_baskets[i] * (1 / self.comp_len))
                sampled_node = np.random.choice(
                    baskets[i], num_sampled_nodes, replace=False
                )
                baskets[i] = np.setdiff1d(baskets[i], sampled_node)
                sampled_nodes.append(sampled_node)
            sampled_nodes = np.concatenate(sampled_nodes, axis=0)
            indices_history.append(
                torch.from_numpy(sampled_nodes).to(padded_mask.device)
            )
            # print([len(x) for x in baskets])
        return indices_history

    def get_encoded_features(self, x, padded_mask, indices_history, mlps):
        """
        Parameters
        ----------
        x: torch.Tensor: [#activated_nodes, embed_dim]
        indices_history: list of torch.Tensor: [comp_len, #sampled nodes]
        """
        compressed_x_list = []
        for idx in range(self.comp_len):
            # node 정보 처리
            # zeros_x = zeros_x.fill_(0).detach()
            # zeros_nodes = zeros_nodes.fill_(0).detach()
            zeros_x = torch.zeros(
                self.num_times, self.num_tokens, self.comp_dim, device=x.device
            )
            zeros_nodes = torch.zeros(x.shape[0], self.comp_dim, device=x.device)
            # [#activated_nodes, comp_dim]
            indices = indices_history[idx]
            # [#activated nodes, embed_dim] -> [#sampled nodes, embed_dim] -> [#sampled nodes, comp_dim] -> [#activated_nodes, comp_dim]
            zeros_nodes[indices, :] = mlps[idx](x[indices, :])
            # [#activated_nodes, comp_dim] -> [#timestamps, #tokens (activated nodes + edges), comp_dim]
            zeros_x[padded_mask, :] = zeros_nodes
            # zero 인 token 이 많을수밖에 없고 time 마다 zero 인 token 의 수가 차이가 크므로 avg pooling 에 대해 normalize. 여긴 edge token 이 zero 이고, pad token 도 zero 이고, indices 가 아닌 nodes token 도 zero 임. time 별로 zero 인 token 의 수가 다른 부분은에 대해 normalize factor 를 곱해서 해결
            normalize_factor = (
                (1 / ((zeros_x.detach().mean(2) != 0).float().mean(1) + 1e-15))
                .unsqueeze(1)
                .unsqueeze(1)
            )  # [#timestamps, 1, 1]
            # [#timestamps, #tokens (activated nodes + edges), comp_dim] -> [#timestamps, 1, comp_dim]
            # maxpooling 만 사용하면 node 1 개 정보만 전달되어 다른 정보가 묵살되는 경향을 보임
            zeros_x_comp = torch.nn.functional.adaptive_avg_pool1d(
                zeros_x.transpose(1, 2), 1
            ).transpose(1, 2)
            zeros_x_comp *= normalize_factor
            #
            compressed_x_list.append(zeros_x_comp)

        return compressed_x_list

    def encode(self, x, padded_node_mask, padded_edge_mask):
        """
        Parameters
        ----------
        x: torch.Tensor
            [#timestamps, #tokens (node + edge), embed_dim]
        """
        x = self.encode_layer_norm(x)
        self.num_times = x.shape[0]
        self.num_tokens = x.shape[1]
        # [#timestamps, #tokens (activated nodes + edges), embed_dim] -> [#activated_nodes, embed_dim]
        nodes = x[padded_node_mask, :]
        self.num_activated_nodes = nodes.shape[0]
        # [#timestamps, #tokens (activated nodes  + edges), embed_dim] -> [#edges of activated_nodes, embed_dim]
        edges = x[padded_edge_mask, :]
        self.num_edges = edges.shape[0]

        compressed_x_list = []

        self.node_indices_history = self.get_indices_from_baskets(padded_node_mask)
        self.edge_indices_history = self.get_indices_from_baskets(padded_edge_mask)
        # [#timestamps, #tokens (activated nodes + edges), comp_dim]
        if self.training:
            # shuffle indices_histories
            np.random.shuffle(self.node_indices_history)
            np.random.shuffle(self.edge_indices_history)

        compressed_x_list = self.get_encoded_features(
            nodes, padded_node_mask, self.node_indices_history, self.node_comp_mlps
        )
        compressed_x_list += self.get_encoded_features(
            edges, padded_edge_mask, self.edge_indices_history, self.edge_comp_mlps
        )

        compressed_x = torch.cat(compressed_x_list, dim=2)
        self.ortho_loss = self.orthogonality_loss(*compressed_x_list)

        return compressed_x

    def decode(self, x, padded_node_mask, padded_edge_mask):
        zeros_residual = torch.zeros(
            x.shape[0], self.num_tokens, self.comp_dim, device=x.device
        )
        # [#timestamps, 1, disentangle_dim (nodes + edges)]
        offset = 0
        zeros_nodes = torch.zeros(
            self.num_activated_nodes, self.comp_dim, device=x.device
        )
        for i, indices in enumerate(self.node_indices_history):
            # [#timestamps, 1, disentangle_dim] -> [#timestamps, 1, comp_dim]
            propagated_features = x[:, :, offset : offset + self.comp_dim]
            # [#timestamps, 1, comp_dim] -> [#timestamps, #tokens (activated nodes + edges), comp_dim] -> [#activated_nodes, comp_dim]
            propagated_nodes = propagated_features.broadcast_to(
                x.shape[0], self.num_tokens, self.comp_dim
            )[padded_node_mask, :]
            # [#activated_nodes, comp_dim] -> [#sampled nodes, comp_dim] -> [#sampled nodes, embed_dim] -> [#activated_nodes, embed_dim]
            zeros_nodes[indices, :] += propagated_nodes[indices, :]
            offset += self.comp_dim

        zeros_residual[padded_node_mask, :] += zeros_nodes

        zeros_edges = torch.zeros(self.num_edges, self.comp_dim, device=x.device)
        # do not set offset to 0
        for i, indices in enumerate(self.edge_indices_history):
            propagated_features = x[:, :, offset : offset + self.comp_dim]
            propagated_edges = propagated_features.broadcast_to(
                x.shape[0], self.num_tokens, self.comp_dim
            )[padded_edge_mask, :]
            zeros_edges[indices, :] += propagated_edges[indices, :]
            offset += self.comp_dim

        zeros_residual[padded_edge_mask, :] += zeros_edges

        return zeros_residual  # self.decode_layer_norm(zeros_residual)

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
