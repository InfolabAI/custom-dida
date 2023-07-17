from trainer import Trainer
from torch_geometric.utils import negative_sampling
from model_DIDA.utils.mutils import *
from tqdm import tqdm
from plot_based_on_hub_nodes import PlotBasedOnHubNodes


class TesterOurs(Trainer):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)
        pass

    def test(self, epoch, data):
        # plot init
        if self.runnerProperty == None:
            raise Exception("You need to set setRunnerProperty first.")

        args = self.runnerProperty.args
        self.model.eval()

        with torch.no_grad():
            cs = []
            with tqdm(total=len(data), leave=False) as pbar:
                pbar.set_description(f"Test")
                for t in range(len(data)):
                    z = self.model(data, t, epoch, is_train=False)
                    cs += [z]
                    # pbar next step
                    pbar.update(1)

            # test
            val_auc_list = []
            test_auc_list = []
            train_auc_list = []
            with tqdm(total=self.runnerProperty.len - 1, leave=False) as pbar:
                pbar.set_description("Test")
                for t in range(self.runnerProperty.len - 1):
                    z = cs[t]
                    _, pos_edge, neg_edge = self.prepare(t + 1)[:3]
                    auc, ap = self.runnerProperty.loss.predict(
                        z, pos_edge, neg_edge, self.model.cs_decoder
                    )
                    if t < self.runnerProperty.len_train - 1:
                        train_auc_list.append(auc)
                    elif (
                        t
                        < self.runnerProperty.len_train
                        + self.runnerProperty.len_val
                        - 1
                    ):
                        val_auc_list.append(auc)
                    else:
                        test_auc_list.append(auc)

                    pbar.update(1)

        return [
            epoch,
            np.mean(train_auc_list),
            np.mean(val_auc_list),
            np.mean(test_auc_list),
        ]