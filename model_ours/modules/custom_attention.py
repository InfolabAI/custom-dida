import torch
import torch.nn as nn
import math
from torch_scatter import scatter
from .droppath import DropPath
from .feedforward import FeedForward
from loguru import logger
from model_ours.modules.multihead_attention import MultiheadAttention


class CustomMultiheadAttention(MultiheadAttention):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        attention_dropout,
        q_noise,
        qn_block_size,
        activation_fn,
        activation_dropout,
        dropout,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            attention_dropout=attention_dropout,
            self_attention=True,
        )
        self.args = args
        self.load_positional_encoding(args.encoder_embed_dim, 1000, args.device)
        self.drop_path = DropPath(0.1, dim=0)
        self.layer_norm_in = nn.LayerNorm(embed_dim)
        self.layer_norm_out = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x == [#timestamps, #tokens (edge features are removed), embed_dim]
        residual = x
        x = self.layer_norm_in(x)
        # x == [#timestamps, #tokens, embed_dim] -> [#timestamps, 1, embed_dim]
        x = torch.nn.functional.adaptive_avg_pool1d(
            x.transpose(1, 2), 1
        ) + torch.nn.functional.adaptive_max_pool1d(x.transpose(1, 2), 1)
        x = x.transpose(1, 2)
        # positional encoding. self.pe.shape == [max_position, embed_dim] -> [max_position, 1, embed_dim]
        x = x + self.pe[: x.shape[0]].unsqueeze(1)
        # attention map is [#timestamps, #timestamps]
        x, attn = super().forward(x, x, x, attn_bias=None, customize=True)
        # [#timestamps, 1, embed_dim] -> [#timestamps(some elementes are dropped), 1, embed_dim]
        x = self.drop_path(x)
        # [#timestamps, 1, embed_dim] -> [#timestamps, #tokens, embed_dim]
        x = x + residual
        return x

    def load_positional_encoding(self, dim_feature=1, max_position=1000, device="cpu"):
        """
        feature 의 위치를 추론에 포함하기 위해 positional embedding을 계산
        https://github.com/InfolabAI/References/blob/eef3666c88f9c4eb5117a0425652295eca012b0e/models/nezha/modeling_nezha.py#L154

        Args:
            d_model: feature의 dimension (현재 1)
            max_len: 위치의 최대값 (현재 window size)
        """
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_position, dim_feature).float()
        pe.require_grad = False

        position = torch.arange(0, max_position).float().unsqueeze(1)
        div_term = (
            torch.arange(0, dim_feature, 2).float() * -(math.log(10000.0) / dim_feature)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.to(device)

    def _to_entier(self, x, batched_data):
        """
        메모리 문제로 실패
        """
        offset = 0
        t_entire_embeddings = []
        for node_num, activated_indices in zip(
            batched_data["node_num"], batched_data["indices_subnodes"]
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

            # node_features are the same across all the timestamps, so, we use [0]
            t_embedding = self.mlp(self.layer_norm_embed(t_activated_embedding))
            t_entire_embeddings.append(t_embedding)
        return torch.stack(t_entire_embeddings, dim=0)

    def _from_entier(self, x, batched_data):
        """
        메모리 문제로 실패
        """
        node_features = []
        for entier_nodes, activated_indices in zip(x, batched_data["indices_subnodes"]):
            node_features.append(entier_nodes[activated_indices.long()])
        return torch.concat(node_features, dim=0)
