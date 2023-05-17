from model_DIDA.config import args
from model_DIDA.utils.mutils import *
from model_DIDA.utils.data_util import *
from model_DIDA.utils.util import init_logger
from model_TokenGT.dataset_handler_tokengt import test
from torch.utils.tensorboard import SummaryWriter
import warnings
from datetime import datetime


def get_current_datetime():
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_date


warnings.simplefilter("ignore")

# load data
args, data = load_data(args)
# pre-logs
log_dir = args.log_dir + args.model_h + "_" + args.dataset + "/"
init_logger(prepare_dir(log_dir) + "log.txt")
info_dict = get_arg_dict(args)

# Runner
from runner import Runner
from model_DIDA.model import DGNN
from model_TokenGT.model_tokengt import TokenGTModel

if args.model_h == "tokengt":
    model = TokenGTModel.build_model().to(args.device)
    log_dir += f"layers_{model.tokengt_args.encoder_layers}_heads_{model.tokengt_args.encoder_attention_heads}"
elif args.model_h == "dida":
    model = DGNN(args=args).to(args.device)
else:
    raise Exception("Unknown model: {}".format(args.model))


runner = Runner(args, model, data, writer=SummaryWriter(log_dir))
results = runner.run()

# post-logs
measure_dict = results
info_dict.update(measure_dict)
json.dump(info_dict, open(osp.join(log_dir, "info.json"), "w"))
