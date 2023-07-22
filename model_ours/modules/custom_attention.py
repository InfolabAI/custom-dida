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
        comp_dim = 2
        super().__init__(
            embed_dim,
            ceil(embed_dim / 2),
            attention_dropout=attention_dropout,
            self_attention=True,
        )
        self.args = args
        self.drop_path = DropPath(0.1, dim=0)
        self.load_positional_encoding(comp_dim, 1000, args.device)
        self.scatter_and_gather1 = ScatterAndGather(args, embed_dim, comp_dim)
        self.scatter_and_gather2 = ScatterAndGather(args, embed_dim, embed_dim)
        self.step = 0

    def forward(self, x, batched_data, padded_node_mask, entire_node_feature):
        # if self.step > 20:
        #    breakpoint()

        # x == [#timestamps, #tokens (node + edge), embed_dim]
        residual = x
        self.args.debug_logger.writer.add_histogram(
            "X/0. input.mean(2).mean(0) of [#timestamps, #tokens (node + edge), embed_dim]",
            x.mean(2).mean(0),
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_histogram(
            "X/0. input.mean(2).mean(1) of [#timestamps, #tokens (node + edge), embed_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )
        # [#timestamps, #tokens (node + edge), embed_dim] -> [#actvated nodes for all the time stamps, embed_dim]
        x = x[padded_node_mask, :]
        self.args.debug_logger.writer.add_histogram(
            "X/1. After extracting nodes, x.mean(1) of [#actvated nodes for all the time stamps, embed_dim]",
            x.mean(1),
            self.args.total_step,
        )
        # [#actvated nodes for all the time stamps, embed_dim] -scatter to entire> [#timestamps, #entire nodes, comp_dim]
        tmpx, entire_activated_indices = self.scatter_and_gather1._to_entire(
            x, batched_data, entire_node_feature, use_bd=False
        )
        sorted_indices = tmpx.mean(2).mean(0).sort(descending=True)[1]
        intersection = torch.zeros_like(
            sorted_indices, dtype=torch.bool, device=sorted_indices.device
        )
        for el in entire_activated_indices:
            intersection = intersection | (el == sorted_indices)
        top_entire_activated_indices = sorted_indices[intersection][:100]
        x, _ = self.scatter_and_gather2._to_entire(
            x, batched_data, entire_node_feature, indices=top_entire_activated_indices
        )
        self.args.debug_logger.writer.add_histogram(
            "X/2. After scattering, x.mean(2).mean(1) of [#timestamps, #entire nodes, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_histogram(
            "X/2. After scattering, x.mean(2).mean(0) of [#timestamps, #entire nodes, comp_dim]",
            x.mean(2).mean(0),
            self.args.total_step,
        )
        # positional encoding. self.pe.shape == [max_position, comp_dim] -> [max_position, 1, comp_dim]
        # x = x + self.pe[: x.shape[0]].unsqueeze(1)
        # attention map is [#timestamps, #timestamps]
        x, attn = super().forward(x, x, x, attn_bias=None, customize=True)
        self.args.debug_logger.writer.add_histogram(
            "X/3. After attention, x.mean(2).mean(1) of [#timestamps, #entire nodes, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_histogram(
            "X/3. After attention, x.mean(2).mean(0) of [#timestamps, #entire nodes, comp_dim]",
            x.mean(2).mean(0),
            self.args.total_step,
        )
        # [#timestamps, #entire nodes, comp_dim] -> [#timestamps(some elementes are dropped), #entire nodes, comp_dim]
        # x = self.drop_path(x)

        # [#timestamps, #entire nodes, comp_dim] -gather from entire-> [#actvated nodes for all the time stamps, embed_dim] -indexing-> [#timestamps, #tokens (node + edge), embed_dim]
        ga = self.scatter_and_gather2._from_entire(
            x, batched_data, top_activated_indices=top_entire_activated_indices
        )
        self.args.debug_logger.writer.add_histogram(
            "X/4. After gathering, x.mean(1) of [#actvated nodes for all the time stamps, embed_dim]",
            x.mean(1),
            self.args.total_step,
        )
        # print(
        #    f"ga/residual[padded_node_mask, :] {ga.abs().mean()/residual[padded_node_mask, :].abs().mean():.2f}"
        # )
        if self.training:
            self.step += 1
        residual[padded_node_mask, :] += ga
        self.args.debug_logger.writer.add_histogram(
            "X/5. After intervening nodes to input, input.mean(2).mean(0) of [#timestamps, #tokens (node + edge), embed_dim]",
            residual.mean(2).mean(0),
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_histogram(
            "X/5. After intervening nodes to input, input.mean(2).mean(1) of [#timestamps, #tokens (node + edge), embed_dim]",
            residual.mean(2).mean(1),
            self.args.total_step,
        )
        # reduce x [#timestamps, #entire nodes, comp_dim] -> [#timestamps, #entire nodes, 1] for broadcasting
        return (
            residual,
            None,
            # (entire_node_feature if entire_node_feature is not None else 0)
            # + torch.nn.functional.adaptive_avg_pool1d(x, 1).detach(),
        )  # self.scatter_and_gather.reduce_mlp(x) + entire_node_feature

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
