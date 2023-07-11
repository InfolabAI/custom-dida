import torch, dgl
import utils_TiaRa
from tqdm import tqdm
from augmenter.augmenter import Augmenter


class EdgePropagation(Augmenter):
    def __init__(self, args, data, eps, device):
        super().__init__(data)
        self.args = args
        self.device = device
        self.discount_factor = 0.5
        self.dense = False

        print("current eps is fixed according to the dataset")
        if args.dataset == "collab":
            self.eps = 0.03
        elif args.dataset == "yelp":
            self.eps = 0.2
        elif args.dataset == "bitcoin":
            self.eps = 0
        elif args.dataset == "wikielec":
            self.eps = 0.01
        elif args.dataset == "redditbody":
            self.eps = 0.01
        else:
            raise NotImplementedError

    def _augment(self, dataset):
        new_graphs = []

        At = dataset[0].adj().to(torch.device(self.device))
        At = torch.sparse_coo_tensor(At.indices(), At.val, At.shape, device=At.device)
        A = At.transpose(1, 0).coalesce()
        new_graph = utils_TiaRa.weighted_adjacency_to_graph(A.cpu())
        new_graphs.append(new_graph)

        for graph in tqdm(dataset[1:], desc="augmentation"):
            At_cur = graph.adj().to(torch.device(self.device))
            At_cur = torch.sparse_coo_tensor(
                At_cur.indices(), At_cur.val, At_cur.shape, device=At_cur.device
            )
            At = self.normalize(At.matmul(At_cur))
            At_cur = At_cur + At
            # At = At_cur + At * self.discount_factor
            # # At 의 최대값이 다시 1이 되도록 연산(원래 모든 edge weight 는 1이었음)
            # At = self.normalize(At)
            # print(At.coalesce().values().max())
            # At_cur = At
            # At 를 filter 하면 node 개수가 줄어들 수 있어 At 와 At_cur 의 + 연산이 불가해지므로, At_filtered 로 따로 저장하고, At 는 누적시킴
            At_filtered = self.filter_matrix(At_cur, self.eps, normalize=False)
            A = At_filtered.transpose(1, 0).coalesce()
            new_graph = utils_TiaRa.weighted_adjacency_to_graph(A.cpu())
            new_graphs.append(new_graph)

        return new_graphs
