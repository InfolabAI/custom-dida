import argparse
import torch
import os

parser = argparse.ArgumentParser()
# 1.dataset
parser.add_argument("--dataset", type=str, default="collab", help="datasets")
parser.add_argument("--num_nodes", type=int, default=-1, help="num of nodes")
parser.add_argument("--nfeat", type=int, default=128, help="dim of input feature")
# TODO ANKI [OBNOTE: data shuffle] - add args
parser.add_argument(
    "--shuffled",
    type=int,
    default=0,
    help="if this option is 1, the time stamps are shuffled",
)
# TODO END ANKI
# TODO ANKI [OBNOTE: plot] - add args for plot
parser.add_argument(
    "--plot",
    type=int,
    default=0,
    help="if this option is 1, the stats of datasets are plotted",
)
# TODO END ANKI

# 2-0.plot
parser.add_argument(
    "--plot_hub_nodes",
    type=int,
    default=0,
    help="if this option is 1, the stats based on hub nodes are plotted",
)
parser.add_argument(
    "--plot_graphs_community_detection",
    type=int,
    default=0,
    help="if this option is 1, the graphs based on community detection are drawed",
)
parser.add_argument(
    "--plot_sparsity_mat_cd",
    type=int,
    default=0,
    help="if this option is 1, the sparsity of adj matrix based on community detection are drawed",
)

# 2-1. augmentation
parser.add_argument(
    "--augment",
    type=str,
    default="no",
    help="tiara | no",
)
parser.add_argument(
    "--hidden_augment",
    type=str,
    default="no",
    help="pool | no",
)


# 3.experiments
parser.add_argument(
    "--model", type=str, help="tokengt_nocd | tokengt_cd | tokengt_cdrandom | dida"
)
parser.add_argument(
    "--max_epoch", type=int, default=100, help="number of epochs to train."
)
parser.add_argument("--testlength", type=int, default=3, help="length for test")
parser.add_argument("--device", type=str, default="cpu", help="training device")
parser.add_argument("--device_id", type=str, default="0", help="device id for gpu")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--patience", type=int, default=50, help="patience for early stop")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-7,
    help="weight for L2 loss on basic models.",
)
parser.add_argument("--output_folder", type=str, default="", help="need to be modified")
parser.add_argument(
    "--sampling_times", type=int, default=1, help="negative sampling times"
)
parser.add_argument("--min_epoch", type=int, default=200, help="min epoch")
parser.add_argument("--log_dir", type=str, default="logs/tmp/")
parser.add_argument("--ex_name", type=str, default="test")
parser.add_argument("--nhid", type=int, default=8, help="dim of hidden embedding")  # 8
parser.add_argument("--n_layers", type=int, default=2)
parser.add_argument("--heads", type=int, default=4, help="attention heads.")  # 4
parser.add_argument("--norm", type=int, default=1)
parser.add_argument("--skip", type=int, default=0, help="")  # 1
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate ")
parser.add_argument("--use_RTE", type=int, default=1, help="")
parser.add_argument("--n_intervene", type=int, default=1000, help="")
parser.add_argument("--la_intervene", type=float, default=0.01)
parser.add_argument("--fmask", type=int, default=1)
parser.add_argument("--lin_bias", type=int, default=0)
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--P", type=float, default=0.5)
parser.add_argument("--SIGMA", type=float, default=0.3)
parser.add_argument("--TEST_P", type=float, default=-0.8)
parser.add_argument("--TEST_SIGMA", type=float, default=0.1)
parser.add_argument("--only_causal", type=int, default=0)
parser.add_argument(
    "--intervention_mechanism", type=int, default=0, help="0 st ; 1 spatio; 2 temporal"
)
parser.add_argument("--learns", type=int, default=1)
parser.add_argument("--warm_epoch", type=int, default=0)
parser.add_argument("--use_cfg", type=int, default=1)
parser.add_argument("--log_interval", type=int, default=2, help="")

args = parser.parse_args()

# set the running device
if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda:{}".format(args.device_id))
    print("using gpu:{} to train the model".format(args.device_id))
else:
    args.device = torch.device("cpu")
    print("using cpu to train the model")


def setargs(args, hp):
    for k, v in hp.items():
        setattr(args, k, v)


if args.use_cfg:
    if args.dataset == "collab":
        hp = {
            "n_layers": 2,
            "heads": 4,
            "norm": 1,
            "skip": 0,
            "dropout": 0.0,
            "use_RTE": 1,
            "n_intervene": 1000,
            "la_intervene": 0.01,
            "fmask": 1,
            "lin_bias": 0,
        }
        setargs(args, hp)
    elif args.dataset == "yelp":
        hp = {
            "n_layers": 2,
            "heads": 4,
            "norm": 1,
            "skip": 0,
            "dropout": 0.0,
            "use_RTE": 1,
            "n_intervene": 1000,
            "la_intervene": 0.01,
            "fmask": 1,
            "lin_bias": 0,
        }
        setargs(args, hp)
    elif "synthetic" in args.dataset:
        hp = {
            "n_layers": 2,
            "heads": 2,
            "norm": 1,
            "skip": 1,
            "dropout": 0.0,
            "use_RTE": 1,
            "n_intervene": 1000,
            "la_intervene": 0.1,
            "fmask": 1,
            "lin_bias": 0,
        }
        setargs(args, hp)
    else:
        hp = {
            "n_layers": 2,
            "heads": 4,
            "norm": 1,
            "skip": 0,
            "dropout": 0.0,
            "use_RTE": 1,
            "n_intervene": 1000,
            "la_intervene": 0.01,
            "fmask": 1,
            "lin_bias": 0,
        }
        setargs(args, hp)
        # raise NotImplementedError(f"dataset {args.dataset} not implemented")
