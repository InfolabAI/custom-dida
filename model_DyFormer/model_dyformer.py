from loguru import logger
from .train_inits import *
from .pretrain_models_graph_trans import (
    train_current_time_step as pretrain_train_current_time_step,
)
from .train_models_graph_trans import (
    train_current_time_step as finetune_train_current_time_step,
)
from .trainer_dyformer import TrainerDyFormer
from .tester_dyformer import TesterDyFormer


def get_pretrain_model_path(FLAGS, res_id, seed=None):
    if seed == None:
        seed = FLAGS.seed

    if res_id is None:
        FLAGS.res_id = "Final_%s_%s_seed_%d_time_%d" % (
            FLAGS.model_name,
            FLAGS.dataset,
            seed,
            FLAGS.time_step,
        )
    else:
        FLAGS.res_id = "Final_%s_%s_seed_%d_time_%d_resid_%s" % (
            FLAGS.model_name,
            FLAGS.dataset,
            seed,
            FLAGS.time_step,
            res_id,
        )

    output_dir = "./model_DyFormer/all_logs/{}/{}/{}".format(
        FLAGS.model_name, FLAGS.dataset, FLAGS.res_id
    )
    return os.path.join(output_dir, "best_valid_model_{}.pt".format(FLAGS.dataset))


class DyFormer:
    def __init__(self, args, data_to_prepare):
        model_name = "DyGraphTransformer_two_stream"

        FLAGS = flags()
        FLAGS.seed = args.seed
        FLAGS.num_epoches = args.max_epoch
        FLAGS.num_epoches = 200
        FLAGS.model_name = model_name

        FLAGS = update_args(FLAGS)
        FLAGS.dataset = args.dataset
        FLAGS.max_neighbors = -1
        FLAGS.deterministic_neighbor_sampling = True

        if FLAGS.dataset in ["Enron_16", "Enron_92", "RDS_100"]:
            FLAGS.two_steam_model = False
        else:
            FLAGS.two_steam_model = True

        FLAGS.num_layers = 2
        """
        Setup layers and res_id
        """
        RES_ID = "pretrain_%d_layer_model" % FLAGS.num_layers

        # # load graphs
        # graphs, adjs = load_graphs(FLAGS.dataset)

        FLAGS.force_regen = False

        FLAGS.supervised = False
        FLAGS.supervised_loss = False
        FLAGS.unsupervised_loss = True

        RES_ID = "pretrain_%d_layer_model_v2" % FLAGS.num_layers

        ############################
        ## put args into FLAGS
        ############################
        self.FLAGS = FLAGS
        self.RES_ID = RES_ID
        self.args = args
        self.device = args.device
        self.data_to_prepare = data_to_prepare
        self.trainer = TrainerDyFormer(args, self, data_to_prepare)
        self.tester = TesterDyFormer(args)

    def parameters(self):
        logger.info("DyFormer parameters are not implemented")
        return [torch.tensor(0.0)]

    def train(self, graphs, adjs, t):
        # t-1 은 get_train_time_interval() 의 eval_start_time 을 참고함
        test_pedges = self.data_to_prepare["pedges"][t - 1]
        test_nedges = self.data_to_prepare["nedges"][t - 1]
        self.FLAGS.time_step = t
        best_valid_model_path = get_pretrain_model_path(
            self.FLAGS, self.RES_ID, seed=123
        )
        if os.path.exists(best_valid_model_path) == False:
            print(best_valid_model_path, "does not exist")
            best_valid_model_path = pretrain_train_current_time_step(
                self.FLAGS,
                graphs,
                adjs,
                self.device,
                self.RES_ID,
                test_pedges=test_pedges,
                test_nedges=test_nedges,
            )

        torch.cuda.empty_cache()

        self.FLAGS.force_regen = False

        # finetuning
        self.FLAGS.supervised = True
        self.FLAGS.supervised_loss = True
        self.FLAGS.unsupervised_loss = False

        self.RES_ID = "finetune_%d_layer_model_downstream_v2" % self.FLAGS.num_layers
        epochs_test_result, epochs_val_result = finetune_train_current_time_step(
            self.FLAGS,
            graphs,
            adjs,
            self.device,
            self.RES_ID,
            model_path=best_valid_model_path,
            test_pedges=test_pedges,
            test_nedges=test_nedges,
        )
        torch.cuda.empty_cache()

        return epochs_test_result, epochs_val_result
