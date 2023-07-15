from trainer import Trainer
from torch_geometric.utils import negative_sampling
from model_DIDA.utils.mutils import *
from hook import gradient_hook, forward_hook
import time
from tqdm import tqdm


class TrainerOurs(Trainer):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)
        self.total_step = 0  # For writer
        pass

    def train(self, epoch, data):
        """
        Parameters
        ----------
        We assume that data is a list of dgl graphs
        """
        super().train(data)
        if self.runnerProperty == None:
            raise Exception("You need to set setRunnerProperty first.")

        args = self.runnerProperty.args
        self.model.train()
        optimizer = self.runnerProperty.optimizer

        edge_index = []
        edge_label = []
        epoch_losses = []
        for t in tqdm(
            range(self.runnerProperty.len_train - 1), desc="Training", leave=False
        ):
            z = self.model(data, t, epoch, is_train=True)

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

            criterion = torch.nn.BCELoss()

            def cal_loss(y, label):
                return criterion(y, label)

            cy = self.model.cs_decoder(z, edge_index)

            loss = cal_loss(cy, edge_label)

            st = time.time()
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-5)
            optimizer.step()
            epoch_losses.append(loss.detach().item())
            self.runnerProperty.writer.add_scalar(
                "bottom_layer_weight_mean",
                self.model.attention.attention.q_proj.weight.detach().cpu().mean(),
                self.total_step,
            )
            self.total_step += 1

        average_epoch_loss = np.array(epoch_losses).mean()
        self.runnerProperty.writer.add_scalar("epoch_loss", loss.detach().cpu(), epoch)

        return average_epoch_loss, [], [], []
