from trainer import Trainer
from model_DIDA.utils.inits import prepare
from torch_geometric.utils import negative_sampling
from model_DIDA.utils.mutils import *
from model_TokenGT.dataset_handler_tokengt import TokenGTDataset
from tqdm import tqdm


class Tester_DIDA(Trainer):
    def __init__(self, args, model):
        super().__init__(args, model)
        pass

    def test(self, epoch, data):
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
            _, pos_edge, neg_edge = prepare(data, t + 1)[:3]
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
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]