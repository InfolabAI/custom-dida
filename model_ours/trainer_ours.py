import time
from trainer_and_tester import TrainerAndTester
from torch_geometric.utils import negative_sampling
from utils_main import *
from tqdm import tqdm


class TrainerOurs(TrainerAndTester):
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
        optimizer = self.runnerProperty.scheduler.optimizer
        scheduler = self.runnerProperty.scheduler

        embeddings = self.model(
            data[: self.runnerProperty.len_train - 1], epoch=epoch, is_train=True
        )

        edge_index = []
        edge_label = []
        train_len = self.runnerProperty.len_train - 1

        for t in range(train_len):
            z = embeddings[t]

            pos_edge_index = self.prepare(t + 1)[0]
            if args.dataset == "yelp":
                neg_edge_index = bi_negative_sampling(
                    pos_edge_index, args.num_nodes, args.shift
                )
            else:
                neg_edge_index = negative_sampling(
                    pos_edge_index,
                    num_neg_samples=pos_edge_index.size(1) * args.sampling_times,
                )
            edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            pos_y = z.new_ones(pos_edge_index.size(1)).to(z.device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(z.device)
            edge_label.append(torch.cat([pos_y, neg_y], dim=0))

        edge_label = torch.cat(edge_label, dim=0)
        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(embeddings[0].device)
            for t in range(train_len):
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        cy = cal_y(embeddings, self.model.cs_decoder)
        loss = cal_loss(cy, edge_label)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-5)
        scheduler.step()
        if self.args.loguru_level == "DEBUG":
            for name, param in self.model.named_parameters():
                self.runnerProperty.writer.add_histogram(
                    name, param, self.args.total_step
                )
        self.args.debug_logger.loguru(
            f"GPU usage",
            f"{get_gpu_memory_usage(self.args.device_id)} MiB",
            1000,
        )
        self.args.total_step += 1

        average_epoch_loss = loss.detach().item()
        self.runnerProperty.writer.add_scalar("epoch_loss", loss.detach().cpu(), epoch)

        return average_epoch_loss, [], [], []
