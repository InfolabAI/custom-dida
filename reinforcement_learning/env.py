import math
import torch
from collections import namedtuple

import numpy as np
import pandas as pd

Reward = namedtuple("Reward", ("total", "long", "short"))

EPS = 1e-20


class DataGenerator:
    def __init__(self, batch_size, dataset, model, runnerProperty):
        """
        Parameters
        ----------
        dataset: we assume that dataset[t] is a input of the model
        """
        self.test_mode = False
        self.dataset = dataset
        self.model = model
        self.runnerProperty = runnerProperty
        # 0번은 이전 graph 가 없는 시작점이라 action 이 의미없으므로 제외하고, episode 의 길이는 1로 고정이므로 len(dataset) - 0 까지 진행
        self.train_indices = np.arange(1, runnerProperty.val_idx)
        self.val_indices = np.arange(runnerProperty.val_idx, runnerProperty.test_idx)
        self.test_indices = np.arange(runnerProperty.test_idx, len(dataset))
        self.batch_size = batch_size

    def _get_initial_embeddings(self):
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                embedding = self.model(self.dataset[idx])
                # TODO change dimension of embedding to build a token
                embeddings.append(embedding)
        return embeddings

    def _get_a_batch(self, indices):
        """
        Parameters
        ----------
        indices: list of int: indices for a batch
        """
        states = []
        embeddings = self._get_initial_embeddings()
        for idx in indices:
            states.append(embeddings[:idx])
            # TODO concat states

        return states

    def _step(self):
        pass

    def reset(self, start_point=None):
        """
        :param start_point:
        :return: obs:(batch, num_assets, window_len, in_features)
        """
        # randomly sample indices from np.array
        if self.test_mode:
            pass
        elif self.val_mode:
            pass
        else:
            cursor = np.random.choice(self.train_indices, self.batch_size)

        return self._get_a_batch(cursor)

    def eval(self):
        self.val_mode = True

    def test(self):
        self.test_mode = True
        self.val_mode = False

    def train(self):
        self.test_mode = False
        self.val_mode = False


class Sim(object):
    def __init__(self):
        pass

    def _step(self, action):
        """
        Caculate reward, info, done(not needed for one-step MDP)
        """
        # reward 계산 시, 추가된 edge 숫자에 대해 페넕티 부과하기
        return  # reward, info, done
        # info = {
        #    "rate_of_return": rate_of_return,
        #    "reward": reward,
        #    "total_value": dv1,
        #    "market_avg_return": market_avg_return,
        #    "weights": w0,
        #    "p": p,
        #    "market_fluctuation": ror,
        # }

    def reset(self):
        pass


class Env(object):
    def __init__(
        self,
    ):
        self.src = DataGenerator()
        self.sim = Sim()

    def step(self, action):
        self.sim._step(action)  # get rewards, info, done
        return

    def reset(self):
        state = self.src.reset()
        return state

    def set_eval(self):
        self.src.eval()

    def set_test(self):
        self.src.test()

    def set_train(self):
        self.src.train()
