import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .disentangler import Disentangler
from math import ceil
from .droppath import DropPath
from loguru import logger
from model_ours.modules.multihead_attention import MultiheadAttention


# funtion for Pytorch's backpropagation hook
def tensor_hook(grad):
    logger.info(f"grad: {grad}")
    return grad


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

    def _sample_from_hard_vectors(self, num_sample):
        # sample hard vector (one-hot vector)
        mask = F.gumbel_softmax(
            torch.stack([self.sample_weights for _ in range(num_sample * 2)]),
            tau=self.args.tau,
            hard=True,
            dim=-1,
        )
        # one-hot vector [num_sample, #subnodes] 에서 [num_sample] dim 이 중복되는 경우 mask 는 1 이상이 됨. 즉, num_sample 보다 sampled_indices 의 수가 적을 수 있고, 중복되는 node 는 mask 로 feature 가 증폭이 됨.
        mask = mask.sum(0)[:num_sample]
        sampled_indices = mask.nonzero().squeeze(1)
        return mask, sampled_indices

    def _sample_from_soft_vectors(self, num_sample, eps=1e-7):
        # sample soft vector
        mask = F.gumbel_softmax(
            self.sample_weights,
            tau=self.args.tau,
            hard=False,
            dim=-1,
        )
        # mask[mask < eps] = 0.0
        sampled_indices = mask.sort(descending=True)[1][:num_sample]
        # sampled_indices = sampled_indices[mask[sampled_indices] > eps]

        # cat_num = ceil(num_sample / sampled_node_features.detach().shape[0])

        return mask, sampled_indices

    def forward(self, x, padded_node_mask, padded_edge_mask, padding_mask):
        x = x.transpose(0, 1)
        status = "train" if self.training else "test"
        self.args.debug_logger.writer.add_histogram(
            f"sample/{status}_sample_weights",
            self.sample_weights.detach(),
            self.args.total_step,
        )

        ######################
        ## generate mask, then sample node features
        pad_mask = ~(padded_node_mask | padded_edge_mask)
        num_sample = pad_mask.sum(1).max()

        mask, sampled_indices = self._sample_from_soft_vectors(num_sample)
        self.args.debug_logger.writer.add_histogram(
            f"sample/{status}_sampled_indices", sampled_indices, self.args.total_step
        )

        sampled_node_features = self.args.batched_data_pool["node_data"][
            sampled_indices
        ] * mask[sampled_indices].unsqueeze(1)

        sampled_original_indices = self.args.batched_data_pool["indices_subnodes"][
            sampled_indices.cpu()
        ]

        ######################
        ## intervene sampled node features to x
        sampled_node_features = sampled_node_features.unsqueeze(0).broadcast_to(
            x.shape[0], -1, -1
        )
        x[pad_mask, :] = sampled_node_features[
            pad_mask[:, pad_mask.shape[1] - num_sample :], :
        ]

        ######################
        ## intervene sampled original indices to self.args.batched_data["indices_subnodes"]
        # 서로 다른 길이를 가진 self.args.batched_data['indices_subnodes'] 에 패딩을 적용해서 stack 하는데 x.shape[1] 만큼의 길이를 가지모도록 torch.ones 추가
        # padding 이 -1 인 이유는 node id 가 0 인 것과 호환성을 위함
        nids = torch.nn.utils.rnn.pad_sequence(
            self.args.batched_data["indices_subnodes"] + [torch.ones(x.shape[1])],
            batch_first=True,
            padding_value=-1,
        ).to(x.device)[:-1]
        sids = torch.stack([sampled_original_indices for _ in range(x.shape[0])]).to(
            x.device
        )
        nids[pad_mask] = sids[pad_mask[:, pad_mask.shape[1] - num_sample :]]
        new_indices_subnodes = []
        for t in range(nids.shape[0]):
            new_indices_subnodes.append(nids[t][nids[t] != -1])

        # 원래 pad 였던 부분이 node 가 되었으므로 True 로 변경
        padded_node_mask[pad_mask] = True
        # Multi-head attention 에서 padding_mask 가 True 인 부분을 무시하는데, padding_mask 가 의미없으므로 None 으로 변경
        padding_mask = None
        # time 별 node 의 수가 바뀌었으므로 변경
        new_node_num = [len(x) for x in new_indices_subnodes]

        return (
            x.transpose(0, 1),
            new_indices_subnodes,
            padded_node_mask,
            padding_mask,
            new_node_num,
        )
