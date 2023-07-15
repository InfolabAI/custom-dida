from subgraph_aware_layers.subgraph_aware import SubgraphAware
import torch.nn.functional as F
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PoolingAttention(SubgraphAware):
    def __init__(self, dida_args):
        super().__init__(dida_args)
        kernel_size = 39
        embed_dim = 1
        reduction_ratio = 1
        # self.conv = nn.Sequential(
        #    nn.Conv1d(
        #        embed_dim,
        #        embed_dim // reduction_ratio,
        #        kernel_size=kernel_size,
        #        padding=(kernel_size - 1) // 2,
        #    ),
        #    # nn.Dropout(0.01),
        #    nn.ReLU(),
        #    nn.Conv1d(
        #        embed_dim // reduction_ratio,
        #        embed_dim // reduction_ratio,
        #        kernel_size=kernel_size,
        #        padding=(kernel_size - 1) // 2,
        #    ),
        #    nn.ReLU(),
        #    nn.Conv1d(
        #        embed_dim // reduction_ratio,
        #        embed_dim,
        #        kernel_size=kernel_size,
        #        padding=(kernel_size - 1) // 2,
        #    ),
        #    nn.ReLU(),
        #    nn.Conv1d(
        #        embed_dim // reduction_ratio,
        #        embed_dim,
        #        kernel_size=kernel_size,
        #        padding=(kernel_size - 1) // 2,
        #    ),
        #    nn.ReLU(),
        #    nn.Conv1d(
        #        embed_dim // reduction_ratio,
        #        embed_dim,
        #        kernel_size=kernel_size,
        #        padding=(kernel_size - 1) // 2,
        #    ),
        # )
        # self.mlp = nn.Sequential(
        #    Flatten(),
        #    nn.Linear(num_maximun_subgraphs, num_maximun_subgraphs // reduction_ratio),
        #    nn.ReLU(),
        #    nn.Linear(num_maximun_subgraphs // reduction_ratio, num_maximun_subgraphs),
        # )
        self.conv2d = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim // reduction_ratio,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                embed_dim // reduction_ratio,
                embed_dim // reduction_ratio,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                embed_dim // reduction_ratio,
                embed_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def _pooling(self, x, pool_type):
        if pool_type == "avg":
            pool = F.avg_pool1d
        elif pool_type == "max":
            pool = F.max_pool1d
        else:
            raise NotImplementedError

        ## For conv1d
        ## T, S, C -> C, S, T
        # attn = x.transpose(0, 2)
        ## C, S, T -> C, S, 1
        # attn = pool(attn, kernel_size=attn.shape[-1])
        ## C, S, 1 -> 1, S, C
        # attn = attn.transpose(0, 2)
        ## C, S, 1 -> 1, S, 1
        # attn = pool(attn, kernel_size=attn.shape[-1])

        # For conv2d
        # T, S, C -> T, S, 1
        attn = pool(x, kernel_size=x.shape[-1])
        # T, S, 1 -> 1, S, T
        attn = attn.transpose(0, 2)

        return attn

    def forward(self, x):
        attn_avg = self._pooling(x, "avg")
        attn_max = self._pooling(x, "max")
        ## For conv1d
        ## (1, S, 1) -> (1, 1, S) -conv-> (1, 1, S) -> (1, S, 1)
        # attn_avg = self.conv(attn_avg.transpose(1, 2)).transpose(1, 2)
        # attn_max = self.conv(attn_max.transpose(1, 2)).transpose(1, 2)

        # For conv2d
        # (1, S, T) -> (1, 1, S, T) -conv-> (1, 1, S, T) -> (1, S, T) -> (T, S, 1)
        attn_avg = self.conv2d(attn_avg.unsqueeze(0)).squeeze(0).transpose(0, 2)
        attn_max = self.conv2d(attn_max.unsqueeze(0)).squeeze(0).transpose(0, 2)

        # mlp (Faild)
        ## convert shape of attn_avg into (1, self.num_maximum_subgraphs)
        # attn_avg = F.pad(
        #    attn_avg.squeeze(2),
        #    pad=(
        #        0,
        #        self.num_maximum_subgraphs - attn_avg.shape[1],
        #        0,
        #        0,
        #    ),  # 4개의 의미는 첫 두개는 shape[1] 에 0~4 범위 padding 추가, 뒤에 두 개는 shape[0] 에 0~0 범위 padding 추가, 즉, 추가 없음
        # )
        ## convert shape of attn_max into (1, self.num_maximum_subgraphs)
        # attn_max = F.pad(
        #    attn_max.squeeze(2),
        #    pad=(
        #        0,
        #        self.num_maximum_subgraphs - attn_max.shape[1],
        #        0,
        #        0,
        #    ),
        # )
        ## after mlp, (1, self.num_maximum_subgraphs) -> (1, 1, self.num_maximum_subgraphs) -> (1, 1, S)
        # attn_avg = F.adaptive_avg_pool1d(self.mlp(attn_avg).unsqueeze(0), x.shape[1])
        # attn_max = F.adaptive_max_pool1d(self.mlp(attn_max).unsqueeze(0), x.shape[1])
        attn = attn_avg + attn_max
        ## (1, 1, S) -> (1, S, 1)
        # attn = attn.transpose(1, 2)
        attn = F.sigmoid(attn)

        return x * attn  # broadcasting
        # return x


"""
normal                      : AUC 0.91
only pooling (1, S, 1)      : AUC 0.91
conv (1, S, C)              : AUC 0.5
conv (1, S, C) with sigmoid : AUC 0.7
conv (1, S, 1)              : AUC 0.92
conv (1, S, 1) with sigmoid : AUC 0.92
conv (1, S, 1) with sigmoid with partitions sorted by size : AUC 0.8
conv2d (T, S, 1) with sigmoid                                : 
mlp (1, S, 1)               : AUC 0.7   
"""
