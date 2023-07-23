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
        optimizer = self.runnerProperty.optimizer
        scheduler = self.runnerProperty.scheduler

        embeddings = self.model(
            data[: self.runnerProperty.len_train - 1], epoch=epoch, is_train=True
        )

        edge_index = []
        edge_label = []
        loss_list = []
        criterion = torch.nn.BCELoss()
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
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
            pos_y = z.new_ones(pos_edge_index.size(1)).to(z.device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(z.device)
            edge_label = torch.cat([pos_y, neg_y], dim=0)

            def cal_loss(y, label):
                return criterion(y, label)

            cy = self.model.cs_decoder(z, edge_index)
            loss = cal_loss(cy, edge_label)
            loss_list.append(loss.unsqueeze(0))

        loss_concat = torch.concat(loss_list, dim=0)
        loss_mean = loss_concat.mean()
        # penalty = torch.nn.functional.mse_loss( loss_concat, torch.zeros_like(loss_concat.detach()))
        penalty = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(loss_concat, dim=0),
            torch.ones_like(loss_concat.detach()) / len(loss_concat),
            reduction="batchmean",
        ) * len(loss_concat)
        loss_total = loss_mean + penalty
        logger.info(f"loss_mean: {loss_mean:.3f}, penalty: {penalty:.3f}")

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        loss_total.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-5)
        scheduler[0].step()
        scheduler[1].step()
        if self.args.loguru_level == "DEBUG" and (epoch % 50 == 0 or epoch == 1):
            for name, param in self.model.named_parameters():
                # self.runnerProperty.writer.add_histogram( name, param, self.args.total_step)
                logger.debug(f"{name} {param.std()}")
        # self.args.debug_logger.loguru( f"GPU usage", f"{get_gpu_memory_usage(self.args.device_id)} MiB", 1000,)

        average_epoch_loss = loss_total.detach().item()
        self.runnerProperty.writer.add_scalar(
            "epoch_loss", loss_total.detach().cpu(), epoch
        )

        return average_epoch_loss, [], [], []
