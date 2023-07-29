import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .disentangler import Disentangler
from math import ceil
from .droppath import DropPath
from loguru import logger
from model_ours.modules.multihead_attention import MultiheadAttention


class InterveneNodes(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        #############################
        ### set sample_weight for gumbel softmax
        self.sample_weights = nn.Parameter(
            torch.ones(
                self.args.batched_data_pool["indices_subnodes"].shape[0],
                device=self.args.device,
            ),
        )
        # from https://github.com/csjunxu/GDAS/blob/5eed8101a78d223a20a43494176051298b24ac3a/lib/nas_rnn/model_search.py#L68, https://github.com/hpnair/18663_Project_FBNet/blob/39a665603e30d83d5997ad78c96a4aa900ba2814/hanna_pytorch/supernet.py#L72
        nn.init.normal_(self.sample_weights, 0, 0.001)
        # nn.init.constant_(x, 1.0)

    def forward(self, x, padded_node_mask, padded_edge_mask):
        x = x.transpose(0, 1)
        self.args.debug_logger.writer.add_histogram(
            "sample/sample_weights", self.sample_weights.detach(), self.args.total_step
        )

        ######################
        ## generate mask, then sample node features
        num_sample = (~padded_node_mask & ~padded_edge_mask).sum(1).max()
        mask = F.gumbel_softmax(
            torch.stack([self.sample_weights for _ in range(num_sample * 2)]),
            tau=self.args.tau,
            hard=True,
            dim=-1,
        )
        # one-hot vector [num_sample, #subnodes] 에서 [num_sample] dim 이 중복되는 경우 mask 는 1 이상이 됨. 즉, num_sample 보다 sampled_indices 의 수가 적을 수 있고, 중복되는 node 는 mask 로 feature 가 증폭이 됨.
        mask = mask.sum(0)[:num_sample]
        sampled_indices = mask.nonzero().squeeze(1)
        self.args.debug_logger.writer.add_histogram(
            "sample/sampled_indices", sampled_indices, self.args.total_step
        )

        sampled_node_features = self.args.batched_data_pool["node_data"][
            sampled_indices
        ] * mask[sampled_indices].unsqueeze(1)

        sampled_original_indices = self.args.batched_data_pool["indices_subnodes"][
            sampled_indices.cpu()
        ]

        ######################
        ## intervene sampled node features to x

        pad_mask = ~(padded_node_mask | padded_edge_mask)
        num_pads = pad_mask.sum()
        cat_num = ceil(num_pads / sampled_node_features.shape[0])
        sampled_node_features_for_T = torch.concat(
            [sampled_node_features for _ in range(cat_num)], dim=0
        )[:num_pads]
        x[pad_mask, :] = sampled_node_features_for_T

        ######################
        ## intervene sampled original indices to self.args.batched_data["indices_subnodes"]
        sampled_original_indices_for_T = torch.concat(
            [sampled_original_indices for _ in range(cat_num)], dim=0
        )[:num_pads]

        offset = 0
        new_indices_subnodes = [[] for _ in range(x.shape[0])]
        for t, num_pad_t in enumerate(pad_mask.sum(1)):
            orig = self.args.batched_data["indices_subnodes"][t]
            sampled = sampled_original_indices_for_T[offset : offset + num_pad_t]
            new_indices_subnodes[t] = torch.concat([orig, sampled])
            offset += num_pad_t

        # 원래 pad 였던 부분이 node 가 되었으므로 True 로 변경
        padded_node_mask[pad_mask] = True
        # time 별 node 의 수가 바뀌었으므로 변경
        new_node_num = [len(x) for x in new_indices_subnodes]

        return x.transpose(0, 1), new_indices_subnodes, padded_node_mask, new_node_num
