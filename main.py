from model_DIDA.config import args
from model_DIDA.utils.mutils import *
from model_DIDA.utils.data_util import *
from model_DIDA.utils.util import init_logger
from model_TokenGT.dataset_handler_tokengt import test
from torch.utils.tensorboard import SummaryWriter
import warnings
from datetime import datetime
from model_TokenGT.visualize_edges import VisualizeEdges
from model_TokenGT.preprocess_raw_data import PreprocessBitcoinAlpha

# TODO ANKI [OBNOTE: ] - what is this?
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# TODO END ANKI


def get_current_datetime():
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date


warnings.simplefilter("ignore")

# plotting
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

# load data
args, data = load_data(args)
# pre-logs
log_dir = f"{args.log_dir}/d{args.is_debug}_{get_current_datetime()}_{args.model_h}_{args.dataset}_s{args.shuffled}_"

# init_logger(prepare_dir(log_dir) + "log.txt")

# Runner
from runner import Runner
from model_DIDA.model import DGNN
from model_TokenGT.model_tokengt import TokenGTModel

try:
    if args.model_h == "tokengt":
        model = TokenGTModel.build_model(args).to(args.device)
        log_dir += f"l{model.tokengt_args.encoder_layers}_h{model.tokengt_args.encoder_attention_heads}"
        prepare_dir(log_dir)
        tokengt_info_dict = get_arg_dict(model.tokengt_args)
        json.dump(tokengt_info_dict, open(osp.join(log_dir, "tokengt_info.json"), "w"))
    elif args.model_h == "dida":
        model = DGNN(args=args).to(args.device)
        prepare_dir(log_dir)
        info_dict = get_arg_dict(args)
        json.dump(info_dict, open(osp.join(log_dir, "info.json"), "w"))
    else:
        raise Exception("Unknown model: {}".format(args.model))

    runner = Runner(args, model, data, writer=SummaryWriter(log_dir))
    results = runner.run()

    # post-logs
    measure_dict = results
    info_dict.update(measure_dict)
    json.dump(info_dict, open(osp.join(log_dir, "info_result.json"), "w"))
except KeyboardInterrupt as e:
    if runner.epoch < 3:
        shutil.rmtree(log_dir)
