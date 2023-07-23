import torch
import torch.nn as nn
import math
import numpy as np
from .scatter_and_gather import ScatterAndGather
from math import ceil
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
            ceil(embed_dim / 2),
            attention_dropout=attention_dropout,
            self_attention=True,
        )
        self.args = args
        self.drop_path = DropPath(0.1, dim=0)
        self.layer_norm_in = nn.LayerNorm(embed_dim)
        self.load_positional_encoding(embed_dim, 1000, args.device)
        self.step = 0

    def forward(self, x):
        # x == [#timestamps, #tokens (edge features are removed), embed_dim]
        residual = x
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/0. input.mean(2).mean(0) of [#timestamps, #tokens (node + edge), embed_dim]",
            x.mean(2).mean(0),
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/0. input.mean(2).mean(1) of [#timestamps, #tokens (node + edge), embed_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )
        x = self.layer_norm_in(x)
        # x == [#timestamps, #tokens, embed_dim] -> [#timestamps, 1, embed_dim]
        residual = x
        x = torch.nn.functional.adaptive_avg_pool1d(
            x.transpose(1, 2), 1
        ) + torch.nn.functional.adaptive_max_pool1d(x.transpose(1, 2), 1)
        x = x.transpose(1, 2)
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/1. After pooling, x.mean(2).mean(1) of [#timestamps, 1, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )
        # positional encoding. self.pe.shape == [max_position, embed_dim] -> [max_position, 1, embed_dim]
        x = x + self.pe[: x.shape[0]].unsqueeze(1)
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/2. After adding positional encoding, x.mean(2).mean(1) of [#timestamps, 1, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )
        # attention map is [#timestamps, #timestamps]
        x, attn = super().forward(x, x, x, attn_bias=None, customize=True)
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/3. After attention, x.mean(2).mean(1) of [#timestamps, 1, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )
        # [#timestamps, 1, embed_dim] -> [#timestamps(some elementes are dropped), 1, embed_dim]
        x = self.drop_path(x)
        # [#timestamps, 1, embed_dim] -> [#timestamps, #tokens, embed_dim]
        x = x + residual
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/4. After intervening nodes to input, input.mean(2).mean(0) of [#timestamps, #tokens (node + edge), embed_dim]",
            residual.mean(2).mean(0),
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/4. After intervening nodes to input, input.mean(2).mean(1) of [#timestamps, #tokens (node + edge), embed_dim]",
            residual.mean(2).mean(1),
            self.args.total_step,
        )
        return x, None

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
