from util_hee import get_current_datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Plot:
    def __init__(
        self,
        args,
        data: dict,
        decoder: nn.Module,
        writer: SummaryWriter,
        epoch: int,
    ):
        """
        Args:
            data: dict with keys "edge_index" "pedges" "nedges"
        """
        assert isinstance(data, dict)
        assert isinstance(decoder, nn.Module)

        self.decoder = decoder
        self.epoch = epoch
        self.args = args
        self.data = data
        self.writer = writer
        # self.log_dir = f"{args.log_dir}/{get_current_datetime()}_plot_{args.model}/"
        pass
