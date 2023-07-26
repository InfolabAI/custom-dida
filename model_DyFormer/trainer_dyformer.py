import time
import networkx as nx
from trainer_and_tester import TrainerAndTester
from utils_main import *


class TrainerDyFormer(TrainerAndTester):
    def __init__(self, args, model, data_to_prepare):
        super().__init__(args, model, data_to_prepare)
        pass

    def train(self, data):
        """
        Parameters
        ----------
        data: list of nx.Graph"""
        # from load_graphs() in preprocess.py of DyFormer
        adjs = list(map(lambda x: nx.adjacency_matrix(x), data))
        t_test_results = []
        t_val_results = []
        # test 시작점부터 evaluation 시작, 즉, test AUC 측정
        for t in range(
            self.runnerProperty.len_train + self.runnerProperty.len_val,
            self.runnerProperty.len - 1,
        ):
            # t += 2 가 필요한 이유:
            # - 우리는 t까지의 graph 로 t+1 edge 을 추론함
            # - DyFormer 는 t-2 까지의 graph 로 t-1 을 추론함
            #    이유 1: t 가 1부터 시작한다고 간주한 입력일 것이라 가정하여 실제 index 와 sync 맞추기 위해 -1
            #    이유 2: 마지막 t 까지 입력일 것이라 가정하여 -1, 즉, 우리처럼 t+1을 예측하는게 아니라 t를 예측하는 것으로 구현함
            t += 2
            epochs_test_result, epochs_val_result = self.model.train(data, adjs, t)
            t_test_results.append(epochs_test_result)
            t_val_results.append(epochs_val_result)
