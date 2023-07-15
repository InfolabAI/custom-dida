import os
import copy
import sys
import time
import torch
import numpy as np
import torch.optim as optim
from model_DIDA.utils.mutils import *
from augmenter.tiara import TiaRa
from augmenter.edge_propagation import EdgePropagation
from augmenter.normal import Normal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from model_DIDA.utils.inits import prepare
from loss import EnvLoss
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
        self.model.tester.setRunnerProperty(
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
        # test(args, self.x, self.data["train"])
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
        max_val_auc = 0
        max_test_auc = 0
        max_train_auc = 0

        if self.args.augment == "no":
            aug = Normal(args, self.data)
            self.data = aug()
        elif self.args.augment == "edgeprop":
            aug = EdgePropagation(
                args,
                self.data,
                None,
                device=f"cuda:{args.device_id}",
            )
            self.data = aug()
        elif self.args.augment == "tiara":
            aug = TiaRa(
                args,
                self.data,
                alpha=0.2,
                beta=0.3,
                eps=0.05,
                K=100,
                symmetric_trick=False,
                device=f"cuda:{args.device_id}",
            )
            self.data = aug()
            pass
        else:
            raise NotImplementedError("augment not implemented")

        # complete checking shuffled data and not shuffled ori_data
        data = self.model.trainer.preprocess_data_per_ex(self.x, self.data)
        ori_data = copy.deepcopy(data)
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                self.epoch = epoch
                t0 = time.time()
                (
                    epoch_losses,
                    train_auc_list,
                    val_auc_list,
                    test_auc_list,
                ) = self.model.trainer.train(epoch, data)
                average_epoch_loss = epoch_losses
                average_train_auc = np.mean(train_auc_list)

                average_val_auc = np.mean(val_auc_list)
                average_test_auc = np.mean(test_auc_list)
                test_results = self.model.tester.test(epoch, ori_data)

                # update the best results.
                if average_val_auc > max_val_auc:
                    max_val_auc = average_val_auc
                    max_test_auc = average_test_auc
                    max_train_auc = average_train_auc

                    metrics = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc".split(
                        ","
                    )
                    measure_dict = dict(
                        zip(
                            metrics,
                            [max_train_auc, max_val_auc, max_test_auc] + test_results,
                        )
                    )

                    patience = 0

                else:
                    patience += 1
                    if epoch > min_epoch and patience > max_patience:
                        break
                bar.set_postfix(
                    {
                        "Epoch": epoch,
                        "Loss": average_epoch_loss,
                        "Time": time.time() - t0,
                        "Train AUC": test_results[1],
                        "Val AUC": test_results[2],
                        "Test AUC": test_results[3],
                    }
                )

                self.writer.add_scalar("Train AUC", test_results[1], epoch)
                self.writer.add_scalar("Val AUC", test_results[2], epoch)
                self.writer.add_scalar("Test AUC", test_results[3], epoch)

        epoch_time = (time.time() - t_total0) / (epoch - 1)
        metrics = (
            [max_train_auc, max_val_auc, max_test_auc] + test_results + [epoch_time]
        )
        metrics_des = "train_auc,val_auc,test_auc,epoch,test_train_auc,test_val_auc,test_test_auc,epoch_time".split(
            ","
        )
        metrics_dict = dict(zip(metrics_des, metrics))
        df = pd.DataFrame([metrics], columns=metrics_des)
        print(df)
        return metrics_dict
