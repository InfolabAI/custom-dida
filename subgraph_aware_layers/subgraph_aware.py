import torch.nn as nn


class SubgraphAware(nn.Module):
    def __init__(self, dida_args):
        super().__init__()
        self.dids_args = dida_args
        pass
