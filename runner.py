import time
import torch.optim as optim
from model_ours.modules.lr import PolynomialDecayLR
from utils_main import *
from loguru import logger
from loss import EnvLoss
from tqdm import tqdm


class RunnerProperty:
    def __init__(
        self,
        args,
        data,
        writer,
        len,
        len_train,
        len_val,
        len_test,
        loss,
        x,
        optimizer,
        scheduler,
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
        self.scheduler = scheduler


def set_optimizer(args, model):
    if args.model == "dida":
        optimizer = optim.Adam(
            [p for n, p in model.named_parameters() if "ss" not in n],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return (optimizer), (None)
    elif args.model == "ours":
        # graph attention model
        optimizer0 = optim.Adam(
            [p for n, p in model.named_parameters() if n.startswith("attention")],
            lr=args.lr_tg,
            weight_decay=0,
        )
        scheduler0 = PolynomialDecayLR(
            optimizer0,
            warmup_updates=args.warmup_updates,
            tot_updates=args.total_num_update,
            lr=args.lr_tg,
            end_lr=args.end_learning_rate,
            power=args.power,
        )
        # main model
        optimizer1 = optim.Adam(
            [p for n, p in model.named_parameters() if not n.startswith("attention")],
            lr=args.lr_tg,
            weight_decay=args.weight_decay_tg,
        )
        scheduler1 = PolynomialDecayLR(
            optimizer1,
            warmup_updates=args.warmup_updates,
            tot_updates=args.total_num_update,
            lr=args.lr_tg,
            end_lr=args.end_learning_rate,
            power=args.power,
        )
        return (optimizer0, optimizer1), (scheduler0, scheduler1)
    else:
        raise NotImplementedError


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
        self.optimizer, self.scheduler = set_optimizer(args, model)
        self.model = model
        runnerProperty = RunnerProperty(
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
            self.scheduler,
        )
        self.model.trainer.setRunnerProperty(runnerProperty)
        self.model.tester.setRunnerProperty(runnerProperty)
        logger.info(
            "total length: {}, test length: {}".format(self.len, args.testlength)
        )

    def run(self):
        args = self.args
        self.conf_opt = None

        # complete checking shuffled data and not shuffled ori_data
        data = self.model.trainer.preprocess_data_per_run(self.data)
        with tqdm(range(1, args.max_epoch + 1)) as bar:
            for epoch in bar:
                self.epoch = epoch
                t0 = time.time()
                epoch_losses = self.model.trainer.train(epoch, data)[0]
                test_results = self.model.tester.test(epoch, data)

                # update the best results.
                bar.set_postfix(
                    {
                        "Epoch": epoch,
                        "Loss": epoch_losses,
                        "Time": time.time() - t0,
                        "Train AUC": test_results[1],
                        "Val AUC": test_results[2],
                        "Test AUC": test_results[3],
                        "lr": self.optimizer[0].param_groups[0]["lr"],
                    }
                )

                self.writer.add_scalar("Train AUC", test_results[1], epoch)
                self.writer.add_scalar("Val AUC", test_results[2], epoch)
                self.writer.add_scalar("Test AUC", test_results[3], epoch)
