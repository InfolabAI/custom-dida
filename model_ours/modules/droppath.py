"""
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True, dim=1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.dim = dim

    def forward(self, x):
        if self.dim == 1:
            x = self._drop_1d_path(x, self.drop_prob, self.training, self.scale_by_keep)
        elif self.dim == 2:
            x = self._drop_2d_path(x, self.drop_prob, self.training, self.scale_by_keep)
        elif self.dim == 0:
            x = self._drop_0d_path(x, self.drop_prob, self.training, self.scale_by_keep)
        else:
            raise ValueError("target must be 0 or 1")
        return x

    def _drop_2d_path(
        self,
        x,
        drop_prob: float = 0.0,
        training: bool = False,
        scale_by_keep: bool = True,
    ):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
        if drop_prob == 0.0 or not training:
            return x
        # x: T x B x C
        keep_prob = 1 - drop_prob
        random_tensor = x.new_empty(1, 1, x.size(2)).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def _drop_1d_path(
        self,
        x,
        drop_prob: float = 0.0,
        training: bool = False,
        scale_by_keep: bool = True,
    ):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
        if drop_prob == 0.0 or not training:
            return x
        # x: T x B x C
        keep_prob = 1 - drop_prob
        random_tensor = x.new_empty(1, x.size(1), 1).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def _drop_0d_path(
        self,
        x,
        drop_prob: float = 0.0,
        training: bool = False,
        scale_by_keep: bool = True,
    ):
        """
        첫 번째 dimension 에 대해 random 하게 dropout 수행
        """
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            mask.div_(keep_prob)
        x = x * mask
        return x
