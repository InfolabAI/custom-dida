from trainer import Trainer
from torch_geometric.utils import negative_sampling
from model_DIDA.utils.mutils import *
from tqdm import tqdm
from plot_based_on_hub_nodes import PlotBasedOnHubNodes


class Tester_DIDA(Trainer):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)

    def test(self, epoch, data):
        # plot init
        plot_ = PlotBasedOnHubNodes(
            self.args, data, self.model.cs_decoder, self.runnerProperty.writer, epoch
        )

        args = self.args

        train_auc_list = []
        val_auc_list = []
        test_auc_list = []
        self.model.eval()
        embeddings, cs, ss = self.model(
            [
                data["edge_index_list"][ix].long().to(args.device)
                for ix in range(self.runnerProperty.len)
            ],
            self.runnerProperty.x,
        )

        for t in range(self.runnerProperty.len - 1):
            z = cs[t]
            _, pos_edge, neg_edge = self.prepare(t + 1)[:3]
            auc, ap = self.runnerProperty.loss.predict(
                z, pos_edge, neg_edge, self.model.cs_decoder
            )
            if t < self.runnerProperty.len_train - 1:
                train_auc_list.append(auc)
            elif t < self.runnerProperty.len_train + self.runnerProperty.len_val - 1:
                val_auc_list.append(auc)
            else:
                test_auc_list.append(auc)
            plot_.process_t(z, t)

        plot_.process_epoch(epoch)

        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]
