import networkx as nx
import dgl
import numpy as np
import torch
import spicy
from loguru import logger
from time import time
from collections import defaultdict


class ConvertGraphTypes:
    def dgl_to_networkx(self, dglG):
        """
        Examples
        --------
        >>> nx_graph.get_edge_data(0,0)
        >>> {'id': 0, 'w': tensor(1.)}
        """
        nx_graph = nx.Graph(dglG.to_networkx(edge_attrs=["w"]))
        return nx_graph

    def networkx_to_dgl(self, nx_graph):
        dgl_graph = dgl.from_networkx(
            nx_graph, edge_attrs=["w"]
        )  # only nx.DiGraph() is supported, not nx.Graph()

        return dgl_graph

    def networkx_to_edge_tensor(self, nx_graph):
        """
        Examples
        --------
        >>> # 주어진 networkx 그래프
        >>> nx_graph = nx.Graph()
        >>> nx_graph.add_edge(0, 1, weight=1)
        >>> nx_graph.add_edge(0, 2, weight=1)
        >>> nx_graph.add_edge(0, 3, weight=1)
        >>> nx_graph.add_edge(1, 2, weight=1)
        >>> nx_graph.add_edge(1, 3, weight=1)
        >>> nx_graph.add_edge(2, 3, weight=1)

        >>> # networkx 그래프를 edge tensor로 변환
        >>> edge_tensor = networkx_to_edge_tensor(nx_graph)
        """
        # networkx 그래프의 엣지 정보를 가져옵니다
        edge_list = list(nx_graph.edges())
        num_edges = len(edge_list)

        # 엣지 리스트를 edge tensor로 변환합니다
        edge_tensor = np.zeros((2, num_edges), dtype=np.int32)
        # convert edge_tensor to tensor.long
        edge_tensor = torch.from_numpy(edge_tensor).long()

        weights_t = {}
        for i in range(num_edges):
            node_i = edge_list[i][0]
            node_j = edge_list[i][1]
            edge_tensor[0][i] = node_i
            edge_tensor[1][i] = node_j
            # undirected graph이므로 엣지의 가중치는 양방향 엣지의 가중치의 평균으로 정의합니다
            weight = (
                float(nx_graph.get_edge_data(node_i, node_j)["w"])
                + float(nx_graph.get_edge_data(node_j, node_i)["w"])
            ) / 2
            weights_t[(int(node_i), int(node_j))] = weight
            weights_t[(int(node_j), int(node_i))] = weight

        return edge_tensor, weights_t

    def edge_tensor_to_networkx(self, edge_tensor, edge_weights, num_nodes=0):
        """
        Parameters
        --------
        num_nodes: int

        Examples
        --------
        >>> # 주어진 edge tensor
        >>> edge_tensor = np.array([[23034, 23034, 1605, 11045, 11045, 11045],
                            [1605, 4350, 23034, 11048, 11047, 11046]])

        >>> # edge tensor를 그래프로 변환
        >>> graph = edge_tensor_to_graph(edge_tensor)
        """
        # edge_tensor의 shape과 데이터를 확인합니다
        num_edges = edge_tensor.shape[1]
        source_nodes = edge_tensor[0]
        dest_nodes = edge_tensor[1]

        # networkx 그래프 객체를 생성합니다
        graph = nx.DiGraph()

        if num_nodes != 0:
            graph.add_nodes_from(list(range(num_nodes)))

        # 각 엣지를 그래프에 추가합니다
        for i in range(num_edges):
            source_node = int(source_nodes[i].numpy())
            dest_node = int(dest_nodes[i].numpy())
            graph.add_edge(
                source_node, dest_node, w=edge_weights[(source_node, dest_node)]
            )

        return graph

    def dict_to_list_of_networkxG(self, data_dict):
        """
        Parameters:
        data_dict: dict: we assume that data_dict.keys() is ['x', 'train', 'test'] and data_dict['train'].keys() is ['edge_index', 'pedges', 'nedges', 'weights']
        """
        node_features = data_dict["x"]

        list_of_networkx_graphs = []
        for t, edge_tensor in enumerate(data_dict["train"]["pedges"]):
            edge_feature_dict = data_dict["train"]["weights"][t]

            # Create a NetworkX graph
            G = nx.MultiGraph()

            # Add nodes with features to the graph
            for node_id, features in enumerate(node_features):
                G.add_node(node_id, w=features)

            # Add edges with features to the graph
            for (src, dst), weight in edge_feature_dict.items():
                G.add_edge(src, dst, w=weight)

            list_of_networkx_graphs.append(G)

        return list_of_networkx_graphs

    def dict_to_list_of_dglG(self, dict, device):
        """
        Parameters
        --------
        dict: dict: we assume that dict.keys() is ['x', 'train', 'test'] and dict['train'].keys() is ['edge_index', 'pedges', 'nedges', 'weights']
        """
        node_features = dict["x"]
        embed_dim = node_features.shape[1]
        num_nodes = node_features.shape[0]

        list_of_dgl_graphs = []
        for t, edge_tensor in enumerate(dict["train"]["pedges"]):
            edge_feature_dict = dict["train"]["weights"][t]
            # we assume that edge_feature_dict means edge weight between nodei and nodej like { (i, j), weight }
            # { (i, j), weight } -> tensor [#edges, embed_dim]
            edge_features = (
                torch.tensor([v for (k1, k2), v in edge_feature_dict.items()])
                .broadcast_to(embed_dim, -1)
                .t()
            )

            dglG = dgl.graph((edge_tensor[0], edge_tensor[1]), num_nodes=num_nodes)
            dglG.ndata["w"] = node_features
            dglG.edata["w"] = edge_features
            list_of_dgl_graphs.append(dglG.to(device))

        return list_of_dgl_graphs

    def dglG_list_to_pool(self, list_of_dgl_graphs):
        comm = self._get_activated_communities(list_of_dgl_graphs)
        new_node_indices = np.unique(
            np.concatenate([activated_nodes for t, activated_nodes in comm.items()])
        )
        tr_input_pool = defaultdict(list)
        tr_input_pool["indices_subnodes"] = torch.Tensor(new_node_indices).int()
        subgraph = dgl.node_subgraph(list_of_dgl_graphs[0], new_node_indices)
        tr_input_pool["node_data"] = subgraph.ndata["w"]
        return tr_input_pool

    def dglG_list_to_TrInputDict(
        self, list_of_dgl_graphs, sampled_original_indices=None
    ):
        device = list_of_dgl_graphs[0].ndata["w"].device
        comm = self._get_activated_communities(list_of_dgl_graphs)
        tr_input = defaultdict(list)
        node_data_index = 0
        for t, activated_nodes in comm.items():
            # t 에 대해 comm 과 list_of_dgl_graphs 를 sync 하여 subgraph 를 생성
            if sampled_original_indices is not None:
                indices = (
                    torch.concat(
                        [torch.tensor(activated_nodes), sampled_original_indices]
                    )
                    .unique()
                    .tolist()
                )
            else:
                indices = activated_nodes
            subgraph = dgl.node_subgraph(list_of_dgl_graphs[t], indices)
            tr_input["indices_subnodes"].append(torch.Tensor(indices).int())

            tr_input["node_data"].append(subgraph.ndata["w"])

            ########################
            ## activated_nodes 에 해당하는 edge 만 사용
            # subgraph 는 node id 가 subgraph 관점으로 변환되어 있으므로, 원래 node id 로 activated_nodes 인지 여부 파악
            original_srcids = subgraph.ndata[dgl.NID][subgraph.edges()[0]]
            srcids_mask = torch.isin(
                original_srcids, torch.tensor(activated_nodes).to(device)
            )
            original_dstids = subgraph.ndata[dgl.NID][subgraph.edges()[1]]
            dstids_mask = torch.isin(
                original_dstids, torch.tensor(activated_nodes).to(device)
            )

            srcids = subgraph.edges()[0][srcids_mask & dstids_mask]
            dstids = subgraph.edges()[1][srcids_mask & dstids_mask]

            # node i, j 의 feature 를 연산하여 edge feature 로 사용
            tr_input["edge_data"].append(
                subgraph.ndata["w"][srcids] * subgraph.ndata["w"][dstids]
            )
            edge_tensor = torch.concat([srcids.unsqueeze(0), dstids.unsqueeze(0)])
            tr_input["edge_index"].append(edge_tensor)
            tr_input["node_num"].append(subgraph.num_nodes())
            tr_input["edge_num"].append(srcids.shape[0])

        tr_input["node_data"] = torch.concat(tr_input["node_data"])
        tr_input["edge_data"] = torch.concat(tr_input["edge_data"])
        tr_input["edge_index"] = torch.concat(tr_input["edge_index"], dim=1)
        tr_input["x"] = list_of_dgl_graphs[0].ndata["w"]

        return tr_input

    def weighted_adjacency_to_graph(self, adj, node_features):
        """
        Parameters
        --------
        node_features: torch.tensor [#nodes, hidden_dim]
        """
        adj = adj.coalesce()
        indices = adj.indices()
        # 0 value 인 edge 모두 제거하도록 indices 추출
        indices_to_be_removed = torch.where(adj.values() == 0)[0]

        graph = dgl.graph(
            (indices[0, :], indices[1, :]), num_nodes=node_features.shape[0]
        )
        graph.ndata["w"] = node_features

        graph = graph.remove_self_loop()
        graph.remove_edges(indices_to_be_removed)
        # graph = dgl.to_simple(graph)

        return graph

    def _get_activated_communities(self, list_of_dgl_graphs):
        comm = {}
        for t, dglG in enumerate(list_of_dgl_graphs):
            activated_node_indices = torch.concat(dglG.edges()).unique().tolist()
            comm[t] = activated_node_indices

        return comm

    def _get_simple_communities(
        self, activated_node_indices: list, num_entire_nodes: int, minnum_nodes: int
    ):
        """
        activated_nodes 전체를 하나의 community 로 만들고, 나머지도 이전 community size 에 맞는 community 로 만듬

        Parameters
        --------
        minnum_nodes: int: deactivated nodes 로 community 를 만들 때 최소한의 node 수
        """
        entire_nodes = np.arange(num_entire_nodes)
        deactivated_nodes = np.setdiff1d(entire_nodes, activated_node_indices)
        comm = {}
        comm[0] = activated_node_indices
        comm_id = 1
        size = max(minnum_nodes * 2, len(activated_node_indices))
        for i in range(0, len(deactivated_nodes), size):
            comm[comm_id] = deactivated_nodes[i : i + size]
            comm_id += 1
        return comm

    def _eig(self, sym_mat):
        sym_mat = sym_mat.to_dense()
        # (sorted) eigenvectors with torch
        eigval, eigvec = torch.linalg.eigh(sym_mat)
        # for eigval, take abs because torch sometimes computes the first eigenvalue approaching 0 from the negative
        eigvec = eigvec.float()  # [N, N (channels)]
        eigval = torch.sort(torch.abs(torch.real(eigval)))[0]  # [N (channels),]
        return eigvec, eigval  # [N, N (channels)]  [N (channels),]

    def _lap_eig(self, dglG, sparse=True):
        dgl_adj = dglG.adj()
        number_of_nodes = dgl_adj.indices().unique().shape[0]
        in_degree = dgl_adj.long().sum(dim=1).view(-1)
        if sparse:
            At = torch.sparse_coo_tensor(
                dgl_adj.indices(), dgl_adj.val, dgl_adj.shape, device=dgl_adj.device
            )
            diagF = self.create_sparse_diag
        else:
            At = dgl_adj.to_dense()
            diagF = torch.diag

        At = At.detach().float()
        in_degree = in_degree.detach().float()
        # Laplacian
        A = At
        N = diagF(torch.clip(in_degree, min=1) ** -0.5)
        L = diagF(in_degree.fill_(1.0)) - torch.matmul(torch.matmul(N, A), N)
        eigvec, eigval = self._eig(L)
        return eigvec, eigval  # [N, N (channels)]  [N (channels),]

    def create_sparse_diag(self, values):
        n = values.shape[0]

        # 대각 행렬의 인덱스 생성 (대각 원소의 위치)
        indices = torch.arange(n, dtype=torch.long, device=values.device)
        indices = torch.stack([indices, indices])

        # COO Tensor 생성
        diag_coo = torch.sparse_coo_tensor(
            indices=indices, values=values, size=(n, n), device=values.device
        )

        return diag_coo
