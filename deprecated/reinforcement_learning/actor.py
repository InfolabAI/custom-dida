import numpy as np
import torch
import torch.nn as nn
from ..model_ours.modules.multihead_attention import MultiheadAttention


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
        scores = self.model(x)
        return scores
