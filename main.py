from config import args
from model_DIDA.utils.mutils import *
from model_DIDA.utils.data_util import *
from model_DIDA.utils.util import init_logger
from model_TokenGT.dataset_handler_tokengt import test
from torch.utils.tensorboard import SummaryWriter
import warnings
from datetime import datetime
from model_TokenGT.visualize_edges import VisualizeEdges
from model_TokenGT.preprocess_raw_data import PreprocessBitcoinAlpha
from util_hee import get_current_datetime
from draw_community_detection import CommunityDetection


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
CommunityDetection(args, args.dataset, data["train"]["pedges"])

# pre-logs
args.log_dir = f"{args.log_dir}/{args.ex_name}/{get_current_datetime()}_{args.model}_{args.dataset}_"

# init_logger(prepare_dir(log_dir) + "log.txt")

# Runner
from runner import Runner
from model_DIDA.model import DGNN
from model_TokenGT.model_tokengt import TokenGTModel

try:
    if args.model == "tokengt":
        model = TokenGTModel.build_model(args).to(args.device)
        args.log_dir += f"{model.tokengt_args.activation_fn}"
        prepare_dir(args.log_dir)
        tokengt_info_dict = get_arg_dict(model.tokengt_args)
        json.dump(
            tokengt_info_dict, open(osp.join(args.log_dir, "tokengt_info.json"), "w")
        )
    elif args.model == "dida":
        model = DGNN(args=args).to(args.device)
        prepare_dir(args.log_dir)
    else:
        raise Exception("Unknown model: {}".format(args.model))

    info_dict = get_arg_dict(args)
    json.dump(info_dict, open(osp.join(args.log_dir, "info.json"), "w"))

    runner = Runner(args, model, data, writer=SummaryWriter(args.log_dir))
    results = runner.run()

    # post-logs
    measure_dict = results
    info_dict.update(measure_dict)
    json.dump(info_dict, open(osp.join(args.log_dir, "info_result.json"), "w"))
except KeyboardInterrupt as e:
    if runner.epoch < 3:
        shutil.rmtree(args.log_dir)
