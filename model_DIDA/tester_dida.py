from trainer_and_tester import TrainerAndTester
from utils_main import *


class Tester_DIDA(TrainerAndTester):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)

    def test(self, epoch, data):
        # plot init
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

        return [
            epoch,
            np.mean(train_auc_list),
            np.std(train_auc_list),
            np.mean(val_auc_list),
            np.std(val_auc_list),
            np.mean(test_auc_list),
            np.std(test_auc_list),
        ]
