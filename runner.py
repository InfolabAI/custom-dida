import os
import sys
import time
import torch
import numpy as np
import torch.optim as optim
from model_DIDA.utils.mutils import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model_DIDA.utils.inits import prepare
from model_DIDA.utils.loss import EnvLoss
from tqdm import tqdm
import pandas as pd


class RunnerProperty:
    def __init__(
        self, args, data, writer, len, len_train, len_val, len_test, loss, x, optimizer
    ):
        self.args = args
        self.data = data
        self.writer = writer
        self.len = len
        self.len_train = len_train
        self.len_val = len_val
        self.len_test = len_test
        self.loss = loss
        self.x = x
        self.optimizer = optimizer


class Runner(object):
    def __init__(self, args, model, data, writer=None, **kwargs):
        seed_everything(args.seed)
        self.args = args
        self.data = data
        self.writer = writer
        self.len = len(data["train"]["edge_index_list"])
        self.len_train = self.len - args.testlength - args.vallength
        self.len_val = args.vallength
        self.len_test = args.testlength
        x = data["x"].to(args.device)
        self.x = [x for _ in range(self.len)] if len(x.shape) <= 2 else x
        self.loss = EnvLoss(args)
        self.optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.model = model
        self.model.trainer.setRunnerProperty(
            RunnerProperty(
                self.args,
                self.data,
                self.writer,
                self.len,
                self.len_train,
                self.len_val,
                self.len_test,
                self.loss,
                self.x,
                self.optimizer,
            )
        )
        print("total length: {}, test length: {}".format(self.len, args.testlength))

    def run(self):
        args = self.args

        minloss = 10
        min_epoch = args.min_epoch
        max_patience = args.patience
        patience = 0
        self.conf_opt = None
        # if args.learns:
        #    self.conf_opt = optim.Adam(
        #        [p for n, p in self.model.named_parameters() if "ss" in n],
        #        lr=args.lr,
        #        weight_decay=args.weight_decay,
        #    )

        t_total0 = time.time()
        max_auc = 0
        max_test_auc = 0
        max_train_auc = 0

        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                t0 = time.time()
                (
                    epoch_losses,
                    train_auc_list,
                    val_auc_list,
                    test_auc_list,
                ) = self.model.trainer.train(epoch, self.data["train"])
                average_epoch_loss = epoch_losses
                average_train_auc = np.mean(train_auc_list)

                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)

                # update the best results.
                if average_val_auc > max_auc:
                    max_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc

                    test_results = self.test(epoch, self.data["test"])

                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(
                        ","
                    )
                    measure_dict = dict(
                        zip(
                            metrics,
                            [max_train_auc, max_auc, max_test_auc] + test_results,
                        )
                    )

                    patience = 0

                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                if epoch == 1 or epoch % self.args.log_interval == 0:
                    print(
                        "Epoch:{}, Loss: {:.4f}, Time: {:.3f}".format(
                            epoch, average_epoch_loss, time.time() - t0
                        )
                    )
                    print(
                        f"Current: Epoch:{epoch}, Train AUC:{average_train_auc:.4f}, Val AUC: {average_val_auc:.4f}, Test AUC: {average_test_auc:.4f}"
                    )

                    print(
                        f"Train: Epoch:{test_results[0]}, Train AUC:{max_train_auc:.4f}, Val AUC: {max_auc:.4f}, Test AUC: {max_test_auc:.4f}"
                    )
                    print(
                        f"Test: Epoch:{test_results[0]}, Train AUC:{test_results[1]:.4f}, Val AUC: {test_results[2]:.4f}, Test AUC: {test_results[3]:.4f}"
                    )

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        metrics = [max_train_auc, max_auc, max_test_auc] + test_results + [epoch_time]
        metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(
            ","
        )
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics], columns=metrics_des)
        print(df)
        return metrics_dict

    def test(self, epoch, data):
        args = self.args

        train_auc_list = []

        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        embeddings, cs, ss = self.model(
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.len)
            ],
            self.x,
        )

        for t in range(self.len - 1):
            z = cs[t]
            edge_index, pos_edge, neg_edge = prepare(data, t + 1)[:3]
            if is_empty_edges(neg_edge):
                continue
            auc, ap = self.loss.predict(z, pos_edge, neg_edge, self.model.cs_decoder)
            if t < self.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.len_train + self.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)

        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]
