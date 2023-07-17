import os
import copy
import json
from runner import Runner
from model_DIDA.model import DGNN
from model_ours.model_ours import OurModel
from loguru import logger
from config import args
from utils_main import *
from torch.utils.tensorboard import SummaryWriter
from utils_main import get_current_datetime
from dataset_loader.utils_data import load_data, prepare_dir

# prepare log path
args.log_dir = f"{args.log_dir}/{args.ex_name}/{get_current_datetime()}_{args.model}_{args.dataset}_{args.propagate}_{args.alpha_std}_"

# load data
args, data = load_data(args)
# loss 계산 또는 auc 계산을 위한 원본 데이터 (edges), 즉, not augmented edges
data_to_prepare = copy.deepcopy(data["train"])

# select model
if args.model == "ours":
    model = OurModel(args, data_to_prepare, data["x"].shape[0]).to(args.device)
    args = model.args
elif args.model == "dida":
    model = DGNN(args, data_to_prepare).to(args.device)
else:
    raise Exception("Unknown model: {}".format(args.model))

prepare_dir(args.log_dir)

# save args and a log file
info_dict = get_arg_dict(args)
json.dump(info_dict, open(os.path.join(args.log_dir, "tokengt_info.json"), "w"))
logger.add(
    os.path.join(args.log_dir, "log.log"),
    format="{time}:{file}:{function}:{level}:{message}",
    level="INFO",
)
logger.info("Start")

# get the number of model's parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Number of parameters: {num_params}")

# run evaluaion
runner = Runner(
    args,
    model,
    data,
    writer=SummaryWriter(os.path.join(args.log_dir, f"seed{args.seed}")),
)
runner.run()
