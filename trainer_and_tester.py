import numpy as np
from runner import RunnerProperty
from convert_graph_types import ConvertGraphTypes


class TrainerAndTester:
    def __init__(self, args, model, data_to_prepare):
        """
        Parameters
        ----------
        data_to_prepare: dict: 로스계산 또는 auc 계산을 위한 원본 데이터, 즉, not augmented data
        """
        self.runnerProperty = None
        self.args = args
        self.model = model
        self.data_to_prepare = data_to_prepare

    def preprocess_data_per_run(self, data):
        if self.args.model == "ours":
            self.cgt = ConvertGraphTypes()
            data = self.cgt.dict_to_list_of_dglG(data, self.args.device)
        elif self.args.model == "dida":
            pass
        else:
            raise NotImplementedError(f"args.model: {self.args.model}")

        return data

    def preprocess_data_per_epoch(self, data):
        pass

    def train(self, data):
        self.preprocess_data_per_epoch(data)
        pass

    def setRunnerProperty(self, runnerProperty: RunnerProperty):
        self.runnerProperty = runnerProperty

    def prepare(self, t):
        # obtain adj index
        edge_index = (
            self.data_to_prepare["edge_index_list"][t].long().to(self.args.device)
        )  # torch edge index
        pos_index = (
            self.data_to_prepare["pedges"][t].long().to(self.args.device)
        )  # torch edge index
        neg_index = (
            self.data_to_prepare["nedges"][t].long().to(self.args.device)
        )  # torch edge index
        new_pos_index = 0
        new_neg_index = 0
        nodes = list(np.union1d(pos_index.cpu().numpy(), neg_index.cpu().numpy()))
        weights = None
        return (
            edge_index,
            pos_index,
            neg_index,
            nodes,
            weights,
            new_pos_index,
            new_neg_index,
        )
