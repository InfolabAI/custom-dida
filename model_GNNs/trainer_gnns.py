import time
from trainer_and_tester import TrainerAndTester
from torch_geometric.utils import negative_sampling
from utils_main import *
from tqdm import tqdm


class TrainerGNNs(TrainerAndTester):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)
        pass

    def train(self, epoch, data):
        """
        Parameters
        ----------
        We assume that data is a list of dgl graphs
        """
        super().train(data)
        assert self.runnerProperty != None, "You need to do setRunnerProperty first."

        args = self.runnerProperty.args
        self.model.train()
        optimizer = self.runnerProperty.optimizer
        scheduler = self.runnerProperty.scheduler

        embeddings = self.model(
            data[: self.runnerProperty.len_train - 1],
            start=0,
            end=len(data[: self.runnerProperty.len_train - 1]),
        )

        loss_list = []
        criterion = torch.nn.BCELoss()
        train_len = self.runnerProperty.len_train - 1

        for t in range(train_len):
            z = embeddings[t]
            pos_edge_index = self.prepare(t + 1)[0]
            neg_edge_index = negative_sampling_(
                pos=pos_edge_index,
                num_nodes=args.num_nodes,
                shift=args.shift,
                num_neg_samples=pos_edge_index.size(1) * args.sampling_times,
                data_to_prepare=self.data_to_prepare,
                t=t,
                dataset=args.dataset,
            )
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            pos_y = z.new_ones(pos_edge_index.size(1)).to(z.device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(z.device)
            edge_label = torch.cat([pos_y, neg_y], dim=0)

            cy = self.model.cs_decoder(z, edge_index)
            loss = criterion(cy, edge_label)
            loss_list.append(loss.unsqueeze(0))

        loss_concat = torch.concat(loss_list, dim=0)
        loss_mean = loss_concat.mean()
        loss_total = loss_mean

        optimizer[0].zero_grad()
        loss_total.backward()
        for name, param in self.model.named_parameters():
            self.args.debug_logger.writer.add_histogram(
                "Params/" + name, param.grad, epoch
            )
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        for name, param in self.model.named_parameters():
            self.args.debug_logger.writer.add_histogram(
                "Params/" + name + "_after_clip", param.grad, epoch
            )
        optimizer[0].step()

        average_epoch_loss = loss_total.detach().item()
        self.runnerProperty.writer.add_scalar(
            "epoch_loss", loss_total.detach().cpu(), epoch
        )

        return average_epoch_loss, [], [], []
