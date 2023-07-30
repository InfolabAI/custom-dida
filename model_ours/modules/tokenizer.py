import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .orf import gaussian_orthogonal_random_matrix_batched


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphFeatureTokenizer(nn.Module):
    """
    Compute node and edge features for each node and edge in the graph.
    """

    def __init__(
        self,
        args,
        hidden_dim,
        n_layers,
    ):
        super(GraphFeatureTokenizer, self).__init__()

        self.type_id = True
        self.order_encoder = nn.Embedding(2, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

        self.orf_encoder = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        # from orf.py
        q, r = torch.linalg.qr(
            torch.randn((10, args.num_nodes, hidden_dim), device=args.device),
            mode="reduced",
        )
        # [1, num_entire_nodes, hidden_dim] -> [num_entire_nodes, hidden_dim]
        self.orf = torch.nn.functional.normalize(q, p=2, dim=2)

        self.step = 0

    @staticmethod
    def get_batch(
        node_feature, edge_index, edge_feature, node_num, edge_num, perturb=None
    ):
        """
        :param node_feature: Tensor([sum(node_num), D])
        :param edge_index: LongTensor([2, sum(edge_num)])
        :param edge_feature: Tensor([sum(edge_num), D])
        :param node_num: list
        :param edge_num: list
        :param perturb: Tensor([B, max(node_num), D])
        :return: padded_index: LongTensor([B, T, 2]), padded_feature: Tensor([B, T, D]), padding_mask: BoolTensor([B, T])
        """
        seq_len = [n + e for n, e in zip(node_num, edge_num)]
        b = len(seq_len)
        d = node_feature.size(-1)
        max_len = max(seq_len) + 1
        max_n = max(node_num)
        device = edge_index.device

        token_pos = torch.arange(max_len, device=device)[None, :].expand(
            b, max_len
        )  # [B, T]

        seq_len = torch.tensor(seq_len, device=device, dtype=torch.long)[
            :, None
        ]  # [B, 1]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[
            :, None
        ]  # [B, 1]
        edge_num = torch.tensor(edge_num, device=device, dtype=torch.long)[
            :, None
        ]  # [B, 1]

        node_index = torch.arange(max_n, device=device, dtype=torch.long)[
            None, :
        ].expand(
            b, max_n
        )  # [B, max_n]
        node_index = node_index[None, node_index < node_num].repeat(
            2, 1
        )  # [2, sum(node_num)]

        padded_node_mask = torch.less(token_pos, node_num)
        padded_edge_mask = torch.logical_and(
            torch.greater_equal(token_pos, node_num),
            torch.less(token_pos, node_num + edge_num),
        )

        padded_index = torch.zeros(
            b, max_len, 2, device=device, dtype=torch.long
        )  # [B, T, 2]
        padded_index[padded_node_mask, :] = node_index.t()
        padded_index[padded_edge_mask, :] = edge_index.t()

        if perturb is not None:
            perturb_mask = padded_node_mask[:, :max_n]  # [B, max_n]
            node_feature = node_feature + perturb[perturb_mask].type(
                node_feature.dtype
            )  # [sum(node_num), D]

        padded_feature = torch.zeros(
            b, max_len, d, device=device, dtype=node_feature.dtype
        )  # [B, T, D]
        padded_feature[padded_node_mask, :] = node_feature
        padded_feature[padded_edge_mask, :] = edge_feature

        padding_mask = torch.greater_equal(token_pos, seq_len)  # [B, T]
        return (
            padded_index,
            padded_feature,
            padding_mask,
            padded_node_mask,
            padded_edge_mask,
        )

    @staticmethod
    @torch.no_grad()
    def get_node_mask(node_num, device):
        b = len(node_num)
        max_n = max(node_num)
        node_index = torch.arange(max_n, device=device, dtype=torch.long)[
            None, :
        ].expand(
            b, max_n
        )  # [B, max_n]
        node_num = torch.tensor(node_num, device=device, dtype=torch.long)[
            :, None
        ]  # [B, 1]
        node_mask = torch.less(node_index, node_num)  # [B, max_n]
        return node_mask

    @staticmethod
    @torch.no_grad()
    def get_random_sign_flip(eigvec, node_mask):
        b, max_n = node_mask.size()
        d = eigvec.size(1)

        sign_flip = torch.rand(b, d, device=eigvec.device, dtype=eigvec.dtype)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        sign_flip = sign_flip[:, None, :].expand(b, max_n, d)
        sign_flip = sign_flip[node_mask]
        return sign_flip

    def handle_eigvec(self, eigvec, node_mask, sign_flip):
        if sign_flip and self.training:
            sign_flip = self.get_random_sign_flip(eigvec, node_mask)
            eigvec = eigvec * sign_flip
        else:
            pass
        return eigvec

    @staticmethod
    def get_index_embed(node_id, node_mask, padded_index):
        """
        :param node_id: Tensor([sum(node_num), D])
        :param node_mask: BoolTensor([B, max_n])
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, 2D])
        """
        b, max_n = node_mask.size()
        max_len = padded_index.size(1)
        d = node_id.size(-1)

        padded_node_id = torch.zeros(
            b, max_n, d, device=node_id.device, dtype=node_id.dtype
        )  # [B, max_n, D]
        padded_node_id[node_mask] = node_id

        padded_node_id = padded_node_id[:, :, None, :].expand(b, max_n, 2, d)
        padded_index = padded_index[..., None].expand(b, max_len, 2, d)
        index_embed = padded_node_id.gather(1, padded_index)  # [B, T, 2, D]
        index_embed = index_embed.view(b, max_len, 2 * d)
        return index_embed

    def get_type_embed(self, padded_index):
        """
        :param padded_index: LongTensor([B, T, 2])
        :return: Tensor([B, T, D])
        """
        order = torch.eq(padded_index[..., 0], padded_index[..., 1]).long()  # [B, T]
        order_embed = self.order_encoder(order)
        return order_embed

    def forward(self, batched_data, perturb=None):
        (
            node_data,
            node_num,
            edge_index,
            edge_data,
            edge_num,
            indices_subnodes,
        ) = (
            batched_data["node_data"],
            batched_data["node_num"],
            batched_data["edge_index"],
            batched_data["edge_data"],
            batched_data["edge_num"],
            batched_data["indices_subnodes"],
        )

        node_feature = node_data
        edge_feature = edge_data
        device = node_feature.device
        dtype = node_feature.dtype

        (
            # padded_index: source node index 와 target node index 를 가짐(서로 같으면 node, 다르면 edge)
            padded_index,
            padded_feature,
            padding_mask,
            padded_node_mask,
            padded_edge_mask,
        ) = self.get_batch(
            node_feature, edge_index, edge_feature, node_num, edge_num, perturb
        )
        node_mask = self.get_node_mask(
            node_num, node_feature.device
        )  # [B, max(n_node)]

        # apply orf id
        orf_id_list = []
        for id_tensor in indices_subnodes:
            orf_id_list.append(self.orf[self.step % 10][id_tensor])

        self.step += 1

        # [sum(#nodes), embed_dim]
        orf_node_id = torch.concat(orf_id_list, dim=0)
        # [sum(#nodes), embed_dim] -> [#subgraphs, max(#nodes), 2*embed_dim]
        # pad 부분에도 orf 가 있으나 무의미하게 반복됨
        orf_embed = self.get_index_embed(orf_node_id, node_mask, padded_index)
        # [#subgraphs, max(#nodes), 2*embed_dim] -> [#subgraphs, max(#nodes), embed_dim]
        padded_feature = padded_feature + self.orf_encoder(orf_embed)

        # apply type id
        padded_feature = padded_feature + self.get_type_embed(padded_index)

        # pad 에 대해 0으로 만드는 부분
        padded_feature = padded_feature.masked_fill(padding_mask[..., None], float("0"))

        return (
            padded_feature,
            padding_mask,
            padded_index,
            padded_node_mask,
            padded_edge_mask,
        )  # [B, 2+T, D], [B, 2+T], [B, T, 2]
