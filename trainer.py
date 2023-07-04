from runner import RunnerProperty
import numpy as np
from model_TokenGT.dataset_handler_get import get_data_converter


class Trainer:
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

    # TODO ANKI [OBNOTE: ] - shuffle and preprocess data
    def preprocess_data_per_ex(self, x, data):
        data_handler, data_converter = get_data_converter(self.args)
        if data_handler is not None:
            data_ = data_handler(x, data, self.args)
        else:
            data_ = data

        return data_
        # TODO END ANKI

    def preprocess_data_per_epoch(self, data):
        if self.args.shuffled:
            if self.args.model == "dida":
                for key in data.keys():
                    data[key] = self.shuffle(data[key], key=key)
            elif "tokengt" in self.args.model:
                data.converted_data_list = self.shuffle(data.converted_data_list)

    def train(self, data):
        self.preprocess_data_per_epoch(data)
        pass

    def shuffle(self, edges_with_time_stamps):
        # TODO ANKI [OBNOTE: ] - sample indices without duplication by using numpy
        indices = np.random.choice(
            np.arange(len(edges_with_time_stamps)),
            len(edges_with_time_stamps),
            replace=False,
        )
        # TODO END ANKI
        list_ = []
        for i in indices:
            list_.append(edges_with_time_stamps[i])
        print(f"Shuffled indices. {indices}")

        edges_with_time_stamps = list_
        return edges_with_time_stamps

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
