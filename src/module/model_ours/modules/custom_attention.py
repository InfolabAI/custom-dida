import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .disentangler import Disentangler
from math import ceil
from .droppath import DropPath
from loguru import logger
from .multihead_attention import MultiheadAttention


class CustomMultiheadAttention(MultiheadAttention):
    def __init__(
        self,
        args,
        embed_dim,
        attention_dropout,
    ):
        # nodes 를 몇 개로 나눌 것인가
        comp_len = 30
        # 나누어진 각각의 node basket 에 대해 몇개 씩의 feature 를 추출할 것인가
        comp_dim = 4
        disentangle_dim = comp_dim * comp_len * 2
        super().__init__(
            disentangle_dim,
            # 각 head 는 하나의 comp_dim 에 대해 attention 을 수행함
            comp_len * 2,
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
        self.drop_path0d = DropPath(self.args.time_att_0d_dropout, dim=0)
        self.drop_path2d = DropPath(self.args.time_att_2d_dropout, dim=2)
        self.step = 0

    def forward(
        self,
        x,
        padded_node_mask,
        padded_edge_mask,
        indices_subnodes,
        node_num,
        time_entirenodes_emdim=None,
    ):
        x = x.transpose(0, 1)
        # x == [#timestamps, #tokens (edge features are removed), embed_dim]
        residual = x.transpose(0, 1)

        # x == [#timestamps, #tokens, embed_dim] -> [#timestamps, 1, embed_dim]
        x = self.disentangler.encode(
            x,
            padded_node_mask,
            indices_subnodes,
            node_num,
            padded_edge_mask,
            time_entirenodes_emdim,
        )
        # attention map is [#timestamps, #timestamps]
        x_drop0d = self.drop_path0d(x)
        # x_drop2d = self.drop_path2d(x)
        x, attn = super().forward(x_drop0d, x, x, attn_bias=None, customize=True)
        # [#timestamps, 1, embed_dim] -> [#timestamps(some elementes are dropped), 1, embed_dim]
        # x = self.drop_path(x)
        # [#timestamps, 1, embed_dim] -> [#timestamps, #tokens, embed_dim]
        tee, ttke = self.disentangler.decode(x, padded_node_mask, padded_edge_mask)
        residual = residual if ttke is None else residual + ttke
        return residual.transpose(0, 1), tee
