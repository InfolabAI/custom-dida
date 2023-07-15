import os
import copy
from config import args
from model_DIDA.utils.mutils import *
from model_DIDA.utils.data_util import *
from model_DIDA.utils.util import init_logger
from model_TokenGT.dataset_handler_tokengt_noCD import test
from torch.utils.tensorboard import SummaryWriter
import warnings
from datetime import datetime
from model_TokenGT.visualize_edges import VisualizeEdges
from model_TokenGT.preprocess_raw_data import PreprocessBitcoinAlpha
from util_hee import get_current_datetime
from plot_graph_community_detection import PlotGraphMat

# pre-logs
args.log_dir = f"{args.log_dir}/{args.ex_name}/{get_current_datetime()}_{args.model}_{args.dataset}_{args.augment}_{args.hidden_augment}_{args.alpha_std}_"


# TODO ANKI [OBNOTE: ] - what is this?
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# TODO END ANKI


warnings.simplefilter("ignore")

# plotting
# TODO ANKI [OBNOTE: ] -
if args.plot > 0:
    ve = VisualizeEdges()
    ve.example1()
    pr = PreprocessBitcoinAlpha()
    ve.visualize_edges(
        "BitcoinAlpha-YM",
        ve.build_edgedict_from_dynamic_graph(pr.retrieve_edges()),
    )
    args.dataset = "collab"
    _, data = load_data(args)
    ve.visualize_edges(
        "Collab", ve.build_edgedict_from_dynamic_graph(data["train"]["pedges"])
    )
    args.dataset = "yelp"
    _, data = load_data(args)
    ve.visualize_edges(
        "Yelp", ve.build_edgedict_from_dynamic_graph(data["train"]["pedges"])
    )
    # TODO END ANKI

# load data
args, data = load_data(args)
data_to_prepare = copy.deepcopy(
    data["train"]
)  # loss 계산 또는 auc 계산을 위한 원본 데이터 (edges), 즉, not augmented edges
PlotGraphMat(args, args.dataset, data["train"]["pedges"], data["train"]["weights"])

# init_logger(prepare_dir(log_dir) + "log.txt")

# Runner
from runner import Runner
from model_DIDA.model import DGNN
from model_TokenGT.model_tokengt import TokenGTModel
from model_TokenGT.model_ours import OurModel

try:
    if "tokengt" in args.model:
        model = TokenGTModel.build_model(args, data_to_prepare).to(args.device)
        args.log_dir += f"{model.tokengt_args.activation_fn}"
        prepare_dir(args.log_dir)
        tokengt_info_dict = get_arg_dict(model.tokengt_args)
        json.dump(
            tokengt_info_dict, open(osp.join(args.log_dir, "tokengt_info.json"), "w")
        )
    elif args.model == "ours":
        model = OurModel(args, data_to_prepare, data["x"].shape[0]).to(args.device)
        prepare_dir(args.log_dir)
        tokengt_info_dict = get_arg_dict(model.tokengt_args)
        json.dump(
            tokengt_info_dict, open(osp.join(args.log_dir, "tokengt_info.json"), "w")
        )

    elif args.model == "dida":
        model = DGNN(args, data_to_prepare).to(args.device)
        prepare_dir(args.log_dir)
    else:
        raise Exception("Unknown model: {}".format(args.model))

    # get pytorch model's the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    info_dict = get_arg_dict(args)
    json.dump(info_dict, open(osp.join(args.log_dir, "info.json"), "w"))

    runner = Runner(
        args,
        model,
        data,
        writer=SummaryWriter(os.path.join(args.log_dir, f"seed{args.seed}")),
    )
    results = runner.run()

    # post-logs
    measure_dict = results
    info_dict.update(measure_dict)
    json.dump(info_dict, open(osp.join(args.log_dir, "info_result.json"), "w"))
except KeyboardInterrupt as e:
    if runner.epoch < 3:
        shutil.rmtree(args.log_dir)
