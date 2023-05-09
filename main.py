from model_DIDA.config import args
from model_DIDA.utils.mutils import *
from model_DIDA.utils.data_util import *
from model_DIDA.utils.util import init_logger
import warnings

warnings.simplefilter("ignore")

# load data
args, data = load_data(args)

# pre-logs
log_dir = args.log_dir
init_logger(prepare_dir(log_dir) + "log.txt")
info_dict = get_arg_dict(args)

# Runner
from runner import Runner
from model_DIDA.model import DGNN
from model_TokenGT.model_tokengt import TokenGTModel

model = DGNN(args=args).to(args.device)
# model = TokenGTModel.build_model().to(args.device)
runner = Runner(args, model, data)
results = runner.run()

# post-logs
measure_dict = results
info_dict.update(measure_dict)
json.dump(info_dict, open(osp.join(log_dir, "info.json"), "w"))
