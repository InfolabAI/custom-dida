import networkx as nx
import dgl
import numpy as np
import torch
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

    def dglG_to_TrInputDict(self, dglG, partition, maxnum_communities=5):
        """
        Parameters
        --------
        dglG: dgl.graph
        partition: dict: {nodeid: partition_id} from Community detection
        """
        st = time()
        # convert partition to {partition_id: [nodeid1, nodeid2, ...]}
        new_partition = defaultdict(list)
        for nodeid, partition_id in partition.items():
            new_partition[partition_id] += [nodeid]

        # restrict the number of communities
        tmp_partition = {}
        for i in range(maxnum_communities):
            tmp_partition[i] = []
        for com_id, node_list in dict(
            sorted(new_partition.items(), key=lambda x: len(x[1]), reverse=True)
        ).items():
            minlen_idx = np.argmin(np.array([len(x) for x in tmp_partition.values()]))
            tmp_partition[minlen_idx] += node_list
        new_partition = tmp_partition

        # set maxnum_nodes_per_subgraph
        maxnum_nodes_per_subgraph = np.array(
            [len(x) for x in tmp_partition.values()]
        ).max()

        subgraph_list = []
        remained_nodes = np.arange(dglG.num_nodes())
        node_data_index = 0
        mapping_from_orig_to_subgraphs = {}
        # logger.info(">" * 10, "time for converting partition:", time() - st)

        st = time()
        # extract subgraphs with edge
        for partition_id, indices_subnodes in new_partition.items():
            subgraph = dgl.node_subgraph(dglG, indices_subnodes)
            remained_nodes = np.setdiff1d(remained_nodes, indices_subnodes)
            subgraph_list.append(subgraph)
            for i in indices_subnodes:
                mapping_from_orig_to_subgraphs[i] = [node_data_index]
                node_data_index += 1
        # logger.info(">" * 10, "time for extracting subgraphs with edge:", time() - st)

        st = time()
        # extract subgraphs without edge
        for idx in range(0, len(remained_nodes), maxnum_nodes_per_subgraph):
            indices_subnodes = remained_nodes[idx : idx + maxnum_nodes_per_subgraph]
            subgraph = dgl.node_subgraph(dglG, indices_subnodes)
            subgraph_list.append(subgraph)
            for i in indices_subnodes:
                mapping_from_orig_to_subgraphs[i] = [node_data_index]
                node_data_index += 1
        # logger.info(">" * 10, "time for extracting subgraphs without edge:", time() - st)

        st = time()
        a_graph_at_t = defaultdict(list)
        for subgraph in subgraph_list:
            a_graph_at_t["node_data"].append(subgraph.ndata["w"])
            a_graph_at_t["edge_data"].append(subgraph.edata["w"])
            edge_tensor = torch.concat(
                [subgraph.edges()[0].unsqueeze(0), subgraph.edges()[1].unsqueeze(0)]
            )
            if edge_tensor.shape[1] != 0:
                a_graph_at_t["edge_index"].append(edge_tensor)
            a_graph_at_t["node_num"].append(subgraph.num_nodes())
            a_graph_at_t["edge_num"].append(subgraph.num_edges())
        # logger.info(">" * 10, "time for making a_graph_at_t:", time() - st)

        st = time()
        a_graph_at_t["node_data"] = torch.concat(a_graph_at_t["node_data"])
        a_graph_at_t["edge_data"] = torch.concat(a_graph_at_t["edge_data"])
        a_graph_at_t["edge_index"] = torch.concat(a_graph_at_t["edge_index"], dim=1)
        # logger.info(">" * 10, "concat time:", time() - st)
        a_graph_at_t["mapping_from_orig_to_subgraphs"] = mapping_from_orig_to_subgraphs

        return a_graph_at_t

    def weighted_adjacency_to_graph(self, adj, node_features, edge_number_limitation):
        """
        Parameters
        --------
        node_features: torch.tensor [#nodes, hidden_dim]
        edge_number_limitation: int: If edge_number_limitation is 100, then we will remian only 100 edges with high weights.
        """
        adj = adj.coalesce()
        indices = adj.indices()
        # edge 가중치가 큰 순서대로 edge_number_limitation 개수만큼만 남기기 위해 삭제할 indices 선택
        indices_to_be_removed = adj.values().sort(descending=True)[1][
            int(edge_number_limitation) :
        ]
        graph = dgl.graph(
            (indices[0, :], indices[1, :]), num_nodes=node_features.shape[0]
        )
        graph.ndata["w"] = node_features
        # [#edges] -> [#edges, hidden_dim]
        graph.edata["w"] = adj.values().broadcast_to(node_features.shape[1], -1).t()

        # TODO remove duplicated edges like [i,j] and [j,i]
        graph = graph.remove_self_loop()
        graph.remove_edges(indices_to_be_removed)

        return graph
