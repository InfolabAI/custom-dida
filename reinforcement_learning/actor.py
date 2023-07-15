import numpy as np
import torch
import torch.nn as nn
from ..model_TokenGT.modules.multihead_attention import MultiheadAttention


class RLActor(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_attention_heads,
        attention_dropout,
        dropout,
        q_noise,
        qn_block_size,
        args,
    ):
        super().__init__()
        self.model = MultiheadAttention(
            embed_dim,
            num_attention_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.args = args

    def forward(self, x, deterministic):
        # print(
        #    f"x_a: {x_a.max()} {x_a.min()}, {x_a.mean()}, x_m: {x_m.max()} {x_m.min()}, {x_m.mean()}, masks: {masks.max()} {masks.min()}, {masks.sum()}"
        # )
        scores = self.model(x)
        return scores
