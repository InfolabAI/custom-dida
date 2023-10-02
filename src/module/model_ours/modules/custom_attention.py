import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import ceil
from .droppath import DropPath
from loguru import logger
from .multihead_attention import MultiheadAttention

from .disentangler_v1 import Disentangler as DV1
from .disentangler_v2 import Disentangler as DV2
from .disentangler_v3 import Disentangler as DV3
from .disentangler_v4 import Disentangler as DV4
from .disentangler_v5 import Disentangler as DV5
from .disentangler_v6 import Disentangler as DV6
from .disentangler_v7 import Disentangler as DV7
from .disentangler_v8 import Disentangler as DV8

disentabler_dict = {
    1: DV1,
    2: DV2,
    3: DV3,
    4: DV4,
    5: DV5,
    6: DV6,
    7: DV7,
    8: DV8,
}


class CustomMultiheadAttention(MultiheadAttention):
    def __init__(
        self,
        args,
        embed_dim,
        attention_dropout,
    ):
        # nodes 를 몇 개로 나눌 것인가
        comp_len = args.featureprop["comp_len"]
        # 나누어진 각각의 node basket 에 대해 몇개 씩의 feature 를 추출할 것인가
        comp_dim = args.featureprop["comp_dim"]
        disentangle_dim = comp_dim * comp_len
        super().__init__(
            disentangle_dim,
            # 각 head 는 하나의 comp_dim 에 대해 attention 을 수행함
            comp_len,
            attention_dropout=attention_dropout,
            self_attention=True,
        )
        # code to set integer variable to the name of the var dynamically

        self.disentangler = disentabler_dict[args.featureprop["version"]](
            args, embed_dim, comp_len, comp_dim
        )
        logger.info(f"type(self.disentangler): {type(self.disentangler)}")

        self.args = args
        self.drop_path0d = DropPath(self.args.featureprop["time_att_0d_dropout"], dim=0)
        self.drop_path2d = DropPath(self.args.featureprop["time_att_2d_dropout"], dim=2)
        self.step = 0

    def forward(
        self,
        x,
        padded_node_mask,
        padded_edge_mask,
        indices_subnodes,
        node_num,
    ):
        x = x.transpose(0, 1)
        residual = x

        x = self.disentangler.encode(
            x,
            padded_node_mask,
            indices_subnodes,
            node_num,
            padded_edge_mask,
            None,
        )
        x_drop0d = self.drop_path0d(x)
        x_drop2d = self.drop_path2d(x)
        x, attn = super().forward(x_drop0d, x_drop2d, x, attn_bias=None, customize=True)
        ttke = self.disentangler.decode(x, padded_node_mask, padded_edge_mask)

        return (residual + ttke).transpose(0, 1)
