from runner import RunnerProperty
from model_TokenGT.dataset_handler_tokengt import TokenGTDataset
import numpy as np


class Trainer:
    def __init__(self, args, model):
        self.runnerProperty = None
        self.args = args
        self.model = model

    # TODO ANKI [OBNOTE: ] - shuffle and preprocess data
    def preprocess_data_per_ex(self, x, data):
        if self.args.model_h == "tokengt":
            data_ = TokenGTDataset(x, data, self.args.device)
        else:
            data_ = data

        return data_
        # TODO END ANKI

    def preprocess_data_per_epoch(self, data):
        if self.args.shuffled:
            if self.args.model_h == "dida":
                for key in data.keys():
                    data[key] = self.shuffle(data[key], key=key)
            elif self.args.model_h == "tokengt":
                data.converted_data_list = self.shuffle(
                    data.converted_data_list, key="tokengt"
                )

    def train(self, data):
        self.preprocess_data_per_epoch(data)
        pass

    def shuffle(self, edges_with_time_stamps, key):
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
