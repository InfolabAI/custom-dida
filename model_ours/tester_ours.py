from analysis import Analysis
from trainer_and_tester import TrainerAndTester
from utils_main import *
from tqdm import tqdm


class TesterOurs(TrainerAndTester):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)
        pass

    def test(self, epoch, data):
        # plot init
        if self.runnerProperty == None:
            raise Exception("You need to set setRunnerProperty first.")

        self.analysis = Analysis(
            self.args, self.runnerProperty.loss, self.model.cs_decoder
        )
        args = self.runnerProperty.args
        train_auc_list = []
        val_auc_list = []
        test_auc_list = []
        self.model.eval()

        with torch.no_grad():
            embeddings, tr_input = self.model(data, epoch=epoch, is_train=False)

            for t in range(self.runnerProperty.len - 1):
                z = embeddings[t]
                _, pos_edge, neg_edge = self.prepare(t + 1)[:3]
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

                self.analysis.accumulate(pos_edge, neg_edge)

            self.analysis.analysis_activated_node_indices(embeddings, tr_input)

            return [
                epoch,
                np.mean(train_auc_list),
                np.std(train_auc_list),
                np.mean(val_auc_list),
                np.std(val_auc_list),
                np.mean(test_auc_list),
                np.std(test_auc_list),
            ]
