"""
Modified from https://github.com/microsoft/Graphormer
"""

import math
from loguru import logger
from typing import Optional, Tuple

import torch
from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn

from einops import rearrange, repeat

from .performer_pytorch import FastAttention


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        attention_dropout=0.0,
        dropout=0.0,
        bias=True,
        self_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.self_attention = self_attention
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout

        assert self.self_attention, "Only support self attention"
        assert (
            not self.self_attention or self.qkv_same_dim
        ), "Self-attention requires QKV to be of the same size"
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.attention_dropout_module = FairseqDropout(
            attention_dropout, module_name=self.__class__.__name__
        )
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.reset_parameters()
        self.onnx_trace = False

    def performer_finetune_setup(
        self, performer_nb_features, performer_generalized_attention
    ):
        self.fast_attention = FastAttention(
            self.head_dim,
            performer_nb_features,
            causal=False,
            generalized_attention=performer_generalized_attention,
            kernel_fn=nn.ReLU(),
            no_projection=False,
        )
        self.forward = self.forward_performer

    def prepare_for_onnx_export_(self):
        raise NotImplementedError

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def custom_attention(self, attn_probs, tgt_len, bsz, embed_dim):
        x_dim = attn_probs.shape[1]
        # lower triangular matrix [T, T]
        mask = torch.tril(torch.ones(x_dim, x_dim).to(attn_probs.device))
        decay_factor = 0.95
        # 대각선에서 한 칸 멀어질때마다 decay_factor 를 한 번 곱함
        for i in range(1, x_dim):
            decay_mat = (
                torch.tril(torch.ones(x_dim - i, x_dim - i).to(attn_probs.device))
                * decay_factor
            )
            decay_mat[decay_mat == 0] = 1.0
            mask[i:, :-i] *= decay_mat

        # 대각선은 1/T 를 더함(t 자신의 node feature 에 대해서는 attention 보장)
        attn_probs += (
            torch.eye(attn_probs.shape[1], device=attn_probs.device).unsqueeze(0)
            / attn_probs.shape[1]
        )
        attn_probs = attn_probs * mask.unsqueeze(0)
        # 각 row 의 합이 1이 되도록 normalize, 각 row 의 합이 다르면 training 시점과 test 시점의 attention score 의 크긱가 다르게 된다. 각 row 는 t 시점에서의 attention distribution.
        attn_probs = attn_probs * 1 / attn_probs.sum(2, keepdim=True)
        return attn_probs

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        ret_attn_probs: bool = False,
        customize: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        # logger.debug( f"query.shape: {query.shape}, key.shape: {key.shape}, value.shape: {value.shape}")
        q = self.q_proj(query)  # [T, B, D]
        k = self.k_proj(query)  # [T, B, D]
        v = self.v_proj(query)  # [T, B, D]
        q *= self.scaling
        # logger.debug( f"After project, q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # logger.debug( f"After applying multi-heads, q.shape: {q.shape}, k.shape: {k.shape}, v.shape: {v.shape}")
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # logger.debug( f"After bmm(q, k.transpose(1, 2)), attn_weights.shape: {attn_weights.shape}")
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )  # [bsz * num_heads, tgt_len, src_len]
        attn_weights = attn_weights_float.type_as(attn_weights)
        if torch.isnan(attn_weights).any():
            breakpoint()
        attn_probs = self.attention_dropout_module(attn_weights)
        # logger.debug(f"After softmax, attn_probs.shape: {attn_probs.shape}")
        if ret_attn_probs:
            return attn_probs
        if customize:
            attn_probs = self.custom_attention(attn_probs, tgt_len, bsz, embed_dim)

        attn = torch.bmm(attn_probs, v)
        # logger.debug(f"After bmm(attn_probs, v), attn.shape: {attn.shape}")
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)
        # logger.debug(f"After out_proj and dropout, attn.shape: {attn.shape}")

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(
                1, 0
            )  # [num_heads, bsz, tgt_len, src_len]
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

    def forward_performer(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        assert attn_bias is None

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        assert k is not None
        assert k.size(0) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            key_padding_mask = key_padding_mask.to(torch.bool)[:, None, :, None]

        q, k, v = map(
            lambda t: rearrange(t, "n b (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )
        attn = self.fast_attention(q, k, v, key_padding_mask)
        attn = rearrange(attn, "b h n d -> n b (h d)")

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            raise NotImplementedError

        return attn, attn_weights
