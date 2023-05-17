from trainer import Trainer
from model_DIDA.utils.inits import prepare
from torch_geometric.utils import negative_sampling
from model_DIDA.utils.mutils import *
from model_TokenGT.dataset_handler_tokengt import TokenGTDataset
from tqdm import tqdm


class Trainer_TokenGT(Trainer):
    def __init__(self, args, model):
        super().__init__(args, model)
        pass

    def train(self, epoch, data):
        if self.runnerProperty == None:
            raise Exception("You need to set setRunnerProperty first.")

        args = self.runnerProperty.args
        self.model.train()
        optimizer = self.runnerProperty.optimizer
        # conf_opt = self.runnerProperty.conf_opt  # authors do not use this optimizer

        dataset = TokenGTDataset(self.runnerProperty.x, data, args.device)
        with torch.no_grad():
            cs = []
            for index in tqdm(range(len(dataset))):
                batch = dataset[index]
                cs += [self.model(batch)]

            # test
            val_auc_list = []
            test_auc_list = []
            train_auc_list = []
            for t in tqdm(range(self.runnerProperty.len - 1)):
                z = cs[t]
                _, pos_edge, neg_edge = prepare(data, t + 1)[:3]
                auc, ap = self.runnerProperty.loss.predict(
                    z, pos_edge, neg_edge, self.model.cs_decoder
                )
                if t < self.runnerProperty.len_train - 1:
                    train_auc_list.append(auc)
                elif (
                    t < self.runnerProperty.len_train + self.runnerProperty.len_val - 1
                ):
                    val_auc_list.append(auc)
                else:
                    test_auc_list.append(auc)

        # train
        edge_index = []
        edge_label = []
        epoch_losses = []
        ## edge label construction
        with tqdm(total=self.runnerProperty.len_train - 1) as pbar:
            for t in range(self.runnerProperty.len_train - 1):
                pbar.set_description(f"Processing item {t}")

                batch = dataset[t]
                z = self.model(batch)
                pos_edge_index = prepare(data, t + 1)[0]
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

                def cal_y(embeddings, decoder):
                    preds = torch.tensor([]).to(z.device)
                    for t in range(self.runnerProperty.len_train - 1):
                        z = embeddings[t]
                        pred = decoder(z, edge_index[t])
                        preds = torch.cat([preds, pred])
                    return preds

                criterion = torch.nn.BCELoss()

                def cal_loss(y, label):
                    return criterion(y, label)

                cy = self.model.cs_decoder(z, edge_index)

                loss = cal_loss(cy, edge_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
                # print(loss)

        average_epoch_loss = np.array(epoch_losses).mean()

        return average_epoch_loss, train_auc_list, val_auc_list, test_auc_list
