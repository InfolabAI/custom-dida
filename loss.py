import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from config import args
from torch_geometric.utils import negative_sampling
from utils_main import *

device = args.device

EPS = 1e-15
MAX_LOGVAR = 10


class EnvLoss(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    @staticmethod
    def maybe_num_nodes(index, num_nodes=None):
        return index.max().item() + 1 if num_nodes is None else num_nodes

    def decoder(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward(self, z, pos_edge_index, neg_edge_index=None, decoder=None):
        if not decoder:
            decoder = self.decoder
        pos_loss = -torch.log(decoder(z, pos_edge_index) + EPS).mean()
        if neg_edge_index == None:
            args = self.args
            if args.dataset == "yelp":
                neg_edge_index = bi_negative_sampling(
                    pos_edge_index, args.num_nodes, args.shift
                )
            else:
                neg_edge_index = negative_sampling(
                    pos_edge_index,
                    num_neg_samples=pos_edge_index.size(1) * self.sampling_times,
                )
        neg_loss = -torch.log(1 - decoder(z, neg_edge_index) + EPS).mean()

        return pos_loss + neg_loss

    def predict(self, z, pos_edge_index, neg_edge_index, decoder):
        pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
        neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
        """
        (Pdb) p pos_y
            tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
        (Pdb) p pos_y.shape
            torch.Size([2254])
        (Pdb) p pos_edge_index.shape
            torch.Size([2, 2254])
        (Pdb) p z.shape
            torch.Size([23035, 16])
        """
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = decoder(z, pos_edge_index)
        neg_pred = decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), 0

    def mse_from_pred_to_edge(self, z, edge_index, decoder, pos_neg):
        if pos_neg == "pos":
            y = z.new_ones(edge_index.size(1)).to(device)
        elif pos_neg == "neg":
            y = z.new_zeros(edge_index.size(1)).to(device)
        else:
            raise Exception("pos_neg should be pos or neg")
        pred = decoder(z, edge_index)
        y, pred = y.detach().cpu().float(), pred.detach().cpu().float()
        return torch.nn.functional.mse_loss(y, pred)
