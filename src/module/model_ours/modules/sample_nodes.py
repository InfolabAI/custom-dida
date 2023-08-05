import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.num = 100
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

    def _sample_from_hard_vectors(self, num_try):
        # sample hard vector (one-hot vector)
        mask = F.gumbel_softmax(
            torch.stack([self.sample_weights for _ in range(num_try)]),
            tau=self.args.tau,
            hard=True,
            dim=-1,
        )
        # one-hot vector [num_sample, #subnodes] 에서 [num_sample] dim 이 중복되는 경우 mask 는 1 이상이 됨. 즉, num_sample 보다 sampled_indices 의 수가 적을 수 있고, 중복되는 node 는 mask 로 feature 가 증폭이 됨.
        duplicated_sampled_indices = mask.nonzero()[:, 1:].squeeze(1)
        self.args.debug_logger.writer.add_histogram(
            f"sample/{self.training}_sampled_indices",
            duplicated_sampled_indices,
            self.args.total_step,
            bins=[x - 0.5 for x in range(self.sample_weights.shape[0] + 1)],
        )
        mask = mask.mean(0)
        sampled_indices = duplicated_sampled_indices.unique()
        return mask, sampled_indices

    def _sample_from_soft_vectors(self, num_sample, eps=1e-5):
        # sample soft vector
        mask = F.gumbel_softmax(
            self.sample_weights,
            tau=self.args.tau,
            hard=False,
            dim=-1,
        )
        sampled_indices = mask.sort(descending=True)[1][:num_sample]
        sampled_indices = sampled_indices[mask[sampled_indices] > eps]
        mask = mask.clone()
        # eps 가 너무 낮으면 이 과정에서 grad 가 inf 가 됨. 이런 부분을 해결할 수 있는 것이 hard sample
        mask[sampled_indices] *= 1 / mask[sampled_indices]

        # cat_num = ceil(num_sample / sampled_indices.detach().shape[0])
        # repeat 과 다르게, repeat_interleave 는 element 의 순서를 유지하면서 반복함. e.g., [1, 2, 3] -> [1, 1, 2, 2, 3, 3]
        # sampled_indices = sampled_indices.repeat_interleave(cat_num)[:num_sample]

        return mask, sampled_indices

    def _reset_backprop(self, list_of_dgl_graphs):
        """
        이전 backprop 정보를 제거함"""
        for i in range(len(list_of_dgl_graphs)):
            list_of_dgl_graphs[i].ndata["X"] = list_of_dgl_graphs[i].ndata["X"].detach()

    def forward(self, list_of_dgl_graphs):
        self._reset_backprop(list_of_dgl_graphs)

        self.args.debug_logger.writer.add_histogram(
            f"sample/{self.training}_sample_weights",
            self.sample_weights.detach(),
            self.args.total_step,
        )

        mask, sampled_indices = self._sample_from_hard_vectors(self.num)
        logger.info(f"sampled_indices: {sampled_indices.shape}")

        sampled_original_indices = self.args.batched_data_pool["indices_subnodes"][
            sampled_indices.cpu()
        ]

        # backpropagation 시 grad 전파에 문제가 없도록 하기 위해 clone() 사용
        node_features = list_of_dgl_graphs[0].ndata["X"].clone()
        node_features[sampled_original_indices] = node_features[
            sampled_original_indices
        ] * mask[sampled_indices].unsqueeze(1)

        for i in range(len(list_of_dgl_graphs)):
            list_of_dgl_graphs[i].ndata["X"] = node_features

        return list_of_dgl_graphs, sampled_original_indices
