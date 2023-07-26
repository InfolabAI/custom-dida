from trainer_and_tester import TrainerAndTester
from torch_geometric.utils import negative_sampling
from utils_main import *
from tqdm import tqdm


class Trainer_DIDA(TrainerAndTester):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)
        pass

    def train(self, epoch, data):
        super().train(data)
        assert self.runnerProperty != None, "You need to do setRunnerProperty first."

        args = self.runnerProperty.args
        self.model.train()
        optimizer = self.runnerProperty.optimizer
        # conf_opt = self.runnerProperty.conf_opt  # authors do not use this optimizer

        # Q: what is data["edge_index_list"]?
        # A: data["edge_index_list"] is a list of edge_index tensors, each of which is a 2xN tensor
        # Q: what is data['edge_index_list'][ix]?
        # A: data['edge_index_list'][ix] is a 2xN tensor
        # Q: what is cs?
        # A: cs is a list of tensors, each of which is a 2xN tensor
        # Q: what is difference between cs and ss?
        # A: cs is the output of the causal decoder, ss is the output of the spatial decoder
        embeddings, cs, ss = self.model(
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.runnerProperty.len)
            ],
            self.runnerProperty.x,
        )
        device = cs[0].device
        ss = [s.detach() for s in ss]

        # test
        val_auc_list = []
        test_auc_list = []
        train_auc_list = []
        for t in range(self.runnerProperty.len - 1):
            z = cs[t]
            _, pos_edge, neg_edge = self.prepare(t + 1)[:3]
            auc, ap = self.runnerProperty.loss.predict(
                z, pos_edge, neg_edge, self.model.cs_decoder
            )
            if t < self.runnerProperty.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.runnerProperty.len_train + self.runnerProperty.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        # train
        edge_index = []
        edge_label = []
        epoch_losses = []
        tsize = []
        ## edge label construction
        for t in range(self.runnerProperty.len_train - 1):
            z = embeddings[t]  # only used for obtaining dimension
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
            edge_index.append(torch.cat([pos_edge_index, neg_edge_index], dim=-1))
            pos_y = z.new_ones(pos_edge_index.size(1)).to(device)
            neg_y = z.new_zeros(neg_edge_index.size(1)).to(device)
            edge_label.append(torch.cat([pos_y, neg_y], dim=0))
            tsize.append(pos_edge_index.shape[1] * 2)

        edge_label = torch.cat(edge_label, dim=0)

        def cal_y(embeddings, decoder):
            preds = torch.tensor([]).to(device)
            for t in range(self.runnerProperty.len_train - 1):
                z = embeddings[t]
                pred = decoder(z, edge_index[t])
                preds = torch.cat([preds, pred])
            return preds

        criterion = torch.nn.BCELoss()

        def cal_loss(y, label):
            return criterion(y, label)

        cy = cal_y(cs, self.model.cs_decoder)
        sy = cal_y(ss, self.model.ss_decoder)

        conf_loss = cal_loss(sy, edge_label)
        causal_loss = cal_loss(cy, edge_label)

        env_loss = torch.tensor([]).to(device)
        intervention_times = args.n_intervene
        la = args.la_intervene

        if epoch < args.warm_epoch:
            la = 0

        if intervention_times > 0 and la > 0:
            if args.intervention_mechanism == 0:
                # slower version of spatial-temporal
                for i in range(intervention_times):
                    s1 = np.random.randint(len(sy))
                    s = torch.sigmoid(sy[s1]).detach()
                    conf = s * cy
                    # conf=self.model.comb_pred(cs,)
                    env_loss = torch.cat(
                        [env_loss, cal_loss(conf, edge_label).unsqueeze(0)]
                    )
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 1:
                # only spatial
                sy = torch.sigmoid(sy).detach().split(tsize)
                cy = cy.split(tsize)
                for i in range(intervention_times):
                    conf = []
                    for j, t in enumerate(tsize):
                        s1 = np.random.randint(len(sy[j]))
                        s1 = sy[j][s1]
                        conf.append(cy[j] * s1)
                    conf = torch.cat(conf, dim=0)
                    env_loss = torch.cat(
                        [env_loss, cal_loss(conf, edge_label).unsqueeze(0)]
                    )
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 2:
                # only temporal
                alle = torch.cat(edge_index, dim=-1)
                v, idxs = torch.sort(alle[0])
                c = v.bincount()
                tsize = c[c.nonzero()].flatten().tolist()

                sy = torch.sigmoid(sy[idxs]).detach().split(tsize)
                cy = cy[idxs].split(tsize)
                edge_label = edge_label[idxs].split(tsize)

                crit = torch.nn.BCELoss(reduction="none")
                elosses = []
                for j, t in tqdm(enumerate(tsize)):
                    s1 = torch.randint(len(sy[j]), (intervention_times, 1)).flatten()
                    alls = sy[j][s1].unsqueeze(-1)
                    allc = cy[j].expand(intervention_times, cy[j].shape[0])
                    conf = allc * alls
                    alle = edge_label[j].expand(
                        intervention_times, edge_label[j].shape[0]
                    )
                    env_loss = crit(conf.flatten(), alle.flatten()).view(
                        intervention_times, sy[j].shape[0]
                    )
                    elosses.append(env_loss)
                env_loss = torch.cat(elosses, dim=-1).mean(dim=-1)
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            elif args.intervention_mechanism == 3:
                # faster approximate version of spatial-temporal
                select = torch.randperm(len(sy))[:intervention_times].to(sy.device)
                alls = torch.sigmoid(sy).detach()[select].unsqueeze(-1)  # [I,1]
                allc = cy.expand(intervention_times, cy.shape[0])  # [I,E]
                conf = allc * alls
                alle = edge_label.expand(intervention_times, edge_label.shape[0])
                crit = torch.nn.BCELoss(reduction="none")
                env_loss = crit(conf.flatten(), alle.flatten())
                env_loss = env_loss.view(intervention_times, sy.shape[0]).mean(dim=-1)
                env_mean = env_loss.mean()
                env_var = torch.var(env_loss * intervention_times)
                penalty = env_mean + env_var
            else:
                raise NotImplementedError("intervention type not implemented")
        else:
            penalty = 0

        loss = causal_loss + la * penalty

        optimizer[0].zero_grad()
        loss.backward()
        optimizer[0].step()
        average_epoch_loss = loss.item()

        return average_epoch_loss, train_auc_list, val_auc_list, test_auc_list
