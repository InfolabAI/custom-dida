import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger


# funtion for Pytorch's backpropagation hook
def tensor_hook(grad):
    logger.info(f"grad: {grad.abs().mean()}")
    # code to find nan in torch tensor
    if torch.isnan(grad).any():
        breakpoint()
    return grad


class SampleNodes(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        self.num = 1
        self.num_division = 10
        #############################
        ### set sample_weight for gumbel softmax
        self.sample_weights = nn.Parameter(
            torch.ones(self.num_division, device=self.args.device)
        )
        # from https://github.com/csjunxu/GDAS/blob/5eed8101a78d223a20a43494176051298b24ac3a/lib/nas_rnn/model_search.py#L68, https://github.com/hpnair/18663_Project_FBNet/blob/39a665603e30d83d5997ad78c96a4aa900ba2814/hanna_pytorch/supernet.py#L72
        # nn.init.normal_(self.sample_weights, 0, 0.001)
        nn.init.constant_(self.sample_weights, 1.0)

    def _sample_from_hard_vectors(self, num_try):
        if not self.training:
            return None, int(self.sample_weights.sort()[1][-1])

        # sample hard vector (one-hot vector)
        mask = F.gumbel_softmax(
            self.sample_weights,
            tau=self.args.tau,
            hard=True,
            dim=-1,
        )
        # 하나만 sample 함
        sampled_index = mask.nonzero().squeeze(1)
        return mask, int(sampled_index.cpu())

    def _reset_backprop(self, list_of_dgl_graphs):
        """
        이전 backprop 정보를 제거함"""
        for i in range(len(list_of_dgl_graphs)):
            list_of_dgl_graphs[i].ndata["w"] = list_of_dgl_graphs[i].ndata["w"].detach()

    def forward(self, list_of_dgl_graphs):
        self._reset_backprop(list_of_dgl_graphs)

        self.args.debug_logger.writer.add_histogram(
            f"sample/{self.training}_sample_weights",
            self.sample_weights.detach(),
            self.args.total_step,
        )

        mask, sampled_index = self._sample_from_hard_vectors(self.num)
        self.args.debug_logger.writer.add_histogram(
            f"sample/{self.training}_sampled_indices",
            sampled_index,
            self.args.total_step,
            bins=[x - 0.5 for x in range(self.sample_weights.shape[0] + 1)],
        )
        logger.info(f"{self.training} sampled_index: {sampled_index}")

        sampled_original_indices = torch.from_numpy(
            np.array_split(np.arange(self.args.num_nodes), self.num_division)[
                sampled_index
            ]
        )
        if not self.training:
            return list_of_dgl_graphs, sampled_original_indices

        # backpropagation 시 grad 전파에 문제가 없도록 하기 위해 clone() 사용
        node_features = list_of_dgl_graphs[0].ndata["w"].clone()
        node_features[sampled_original_indices] = node_features[
            sampled_original_indices
        ] * mask[sampled_index].unsqueeze(0).unsqueeze(1)

        for i in range(len(list_of_dgl_graphs)):
            list_of_dgl_graphs[i].ndata["w"] = node_features

        return list_of_dgl_graphs, sampled_original_indices
