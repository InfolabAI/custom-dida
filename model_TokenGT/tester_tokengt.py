from trainer import Trainer
from model_DIDA.utils.inits import prepare
from torch_geometric.utils import negative_sampling
from model_DIDA.utils.mutils import *
from model_TokenGT.dataset_handler_tokengt import TokenGTDataset
from tqdm import tqdm


class Tester_TokenGT(Trainer):
    def __init__(self, args, model):
        super().__init__(args, model)
        pass

    def test(self, epoch, data):
        if self.runnerProperty == None:
            raise Exception("You need to set setRunnerProperty first.")

        args = self.runnerProperty.args
        self.model.eval()

        dataset = TokenGTDataset(self.runnerProperty.x, data, args.device)
        with torch.no_grad():
            cs = []
            for index in tqdm(range(len(dataset))):
                batch = dataset[index]
                cs += [self.model(batch)]

            # test
            val_auc_list = []
            test_auc_list = []
            train_auc_list = []
            for t in tqdm(range(self.runnerProperty.len - 1)):
                z = cs[t]
                _, pos_edge, neg_edge = prepare(data, t + 1)[:3]
                auc, ap = self.runnerProperty.loss.predict(
                    z, pos_edge, neg_edge, self.model.cs_decoder
                )
                if t < self.runnerProperty.len_train - 1:
                    train_auc_list.append(auc)
                elif (
                    t < self.runnerProperty.len_train + self.runnerProperty.len_val - 1
                ):
                    val_auc_list.append(auc)
                else:
                    test_auc_list.append(auc)

        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]
