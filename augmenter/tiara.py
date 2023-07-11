import torch
import utils_TiaRa
from augmenter.augmenter import Augmenter
from tqdm.autonotebook import tqdm
from loguru import logger


class TiaRa(Augmenter):
    """
    Approximate augmentation of temporal random walk diffusion

    Parameters
    ----------
    adjacencies
        list(or iterator) of dgl dataset
    alpha
        restart probability
    beta
        time travel probability
    eps
        filtering threshold
    K
        number of iteration for inverting H at time t
    symmetric_trick
        method to generate normalized symmetric adjacency matrix
    device
        device name
    dense
        use dense adjacency matrix
    verbose
        print additional information

    Returns
    -------
    list of augmented dgl dataset

    Examples
    --------
    >>> tiara = Tiara(alpha, beta, device=device)
    >>> augmented_graphs = tiara(dataset)
    """

    def __init__(
        self,
        data,
        alpha=0.2,
        beta=0.3,
        eps=1e-3,
        K=100,
        symmetric_trick=True,
        device="cuda",
        dense=False,
        verbose=False,
    ):
        assert 0 <= alpha and alpha <= 1
        assert 0 <= beta and beta <= 1
        assert 0 <= alpha + beta and alpha + beta <= 1
        assert 0 < eps

        super().__init__(data)

        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.K = K
        self.symmetric_trick = symmetric_trick
        self.device = device
        self.dense = dense
        self.verbose = verbose

    def _augment(self, dataset):
        N = dataset[0].num_nodes()
        for graph in dataset:
            assert N == graph.num_nodes()

        if self.dense:
            I = torch.eye(N, device=self.device)
        else:
            I = utils_TiaRa.sparse_eye(N, self.device)

        Xt_1 = I
        new_graphs = list()

        for graph in tqdm(dataset, desc="augmentation"):
            # FIX: to gpu
            At = graph.adj().to(torch.device(self.device))
            if self.dense:
                At = At.to_dense()
            # FIX: convert At(SparseMatrix) into I(sparse_coo_tensor)
            At = torch.sparse_coo_tensor(
                At.indices(), At.val, At.shape, device=At.device
            )
            At = At + I
            At = self.normalize(At, ord="row")

            inv_Ht = self.approx_inv_Ht(At, self.alpha, self.beta, self.K)

            Xt = inv_Ht @ (self.alpha * I + self.beta * Xt_1)
            # when alpha/beta is small, small K can lead to a large approximate error in Xt
            # for considering this case, we normalize X so that the column sum of X is 1
            Xt = self.normalize(Xt, ord="col")

            Xt = self.filter_matrix(Xt, self.eps)
            Xt_1 = Xt

            if self.symmetric_trick:
                Xt = (Xt + Xt.transpose(1, 0)) / 2

            if self.dense:
                A = Xt.to_sparse().transpose(0, 1).coalesce()
            else:
                A = Xt.transpose(1, 0).coalesce()

            if self.symmetric_trick:
                ones = torch.ones(A._nnz(), device=self.device)
                A = torch.sparse_coo_tensor(
                    A.indices(), ones, A.shape, device=self.device
                )
                A = self.normalize(A, ord="sym", dense=False)
            else:
                A = self.normalize(A, ord="row", dense=False)

            # to cpu
            new_graph = utils_TiaRa.weighted_adjacency_to_graph(A.cpu())
            new_graphs.append(new_graph)

            if self.verbose:
                logger.info(
                    "number of edge in this time step: {}".format(new_graph.num_edges())
                )

        return new_graphs

    def approx_inv_Ht(self, A, alpha, beta, K=10):
        if self.dense:
            I = torch.eye(A.shape[0], device=self.device)
        else:
            I = utils_TiaRa.sparse_eye(A.shape[0], self.device)

        inv_H_k = I
        c = 1.0 - alpha - beta
        cAT = c * A.transpose(0, 1)

        for i in range(K):
            inv_H_k = I + cAT @ inv_H_k

        return inv_H_k
