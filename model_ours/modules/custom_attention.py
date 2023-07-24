import torch
import torch.nn as nn
import math
from .disentangler import Disentangler
from math import ceil
from .droppath import DropPath
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
        # nodes 를 몇 개로 나눌 것인가
        comp_len = 30
        # 나누어진 각각의 node basket 에 대해 몇개 씩의 feature 를 추출할 것인가
        comp_dim = 4
        disentangle_dim = comp_dim * comp_len
        super().__init__(
            disentangle_dim,
            # 각 head 는 하나의 comp_dim 에 대해 attention 을 수행함
            comp_len,
            attention_dropout=attention_dropout,
            self_attention=True,
        )
        self.disentangler = Disentangler(args, embed_dim, comp_len, comp_dim)
        self.to_embed_dim = nn.Sequential(
            nn.Linear(disentangle_dim, comp_dim * 4),
            nn.GELU(),
            nn.Dropout1d(0.1),
            nn.Linear(comp_dim * 4, embed_dim),
        )

        self.args = args
        self.drop_path = DropPath(0.1, dim=0)
        self.load_positional_encoding(disentangle_dim, 1000, args.device)
        self.step = 0

    def log_input(self, x):
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

    def log_encode(self, x):
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/1. After pooling, x.mean(2).mean(1) of [#timestamps, 1, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )

    def log_pe(self, x):
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/2. After adding positional encoding, x.mean(2).mean(1) of [#timestamps, 1, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )

    def log_att(self, x):
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/3. After attention, x.mean(2).mean(1) of [#timestamps, 1, comp_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )

    def log_output(self, x):
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/4. After intervening nodes to input, input.mean(2).mean(0) of [#timestamps, #tokens (node + edge), embed_dim]",
            x.mean(2).mean(0),
            self.args.total_step,
        )
        self.args.debug_logger.writer.add_histogram(
            f"{self.training}-X/4. After intervening nodes to input, input.mean(2).mean(1) of [#timestamps, #tokens (node + edge), embed_dim]",
            x.mean(2).mean(1),
            self.args.total_step,
        )

    def forward(self, x, padded_node_mask, padded_edge_mask):
        # x == [#timestamps, #tokens (edge features are removed), embed_dim]
        residual = x
        self.log_input(x)
        # x == [#timestamps, #tokens, embed_dim] -> [#timestamps, 1, embed_dim]
        x = self.disentangler.encode(x, padded_node_mask, padded_edge_mask)
        self.log_encode(x)
        # attention map is [#timestamps, #timestamps]
        x, attn = super().forward(x, x, x, attn_bias=None, customize=True)
        self.log_att(x)
        # [#timestamps, 1, embed_dim] -> [#timestamps(some elementes are dropped), 1, embed_dim]
        # x = self.drop_path(x)
        # [#timestamps, 1, embed_dim] -> [#timestamps, #tokens, embed_dim]

        residual = (
            residual
            # 적절한 크기의 영향을 주기위해 여기에 linear 를 배치함 이게 없으면 residual 이 dominate 하여 거의 영향을 못 줌
            # + self.to_embed_dim(
            #    self.disentangler.decode(
            #        x, residual, padded_node_mask, padded_edge_mask
            #    )
            # )
            + self.to_embed_dim(x)
        )
        self.log_output(residual)
        return residual, None

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
