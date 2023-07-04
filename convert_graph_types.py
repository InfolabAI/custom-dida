import networkx as nx
import dgl
import numpy as np
import torch


class ConvertGraphTypes:
    def dgl_to_networkx(self, dglG):
        nx_graph = dglG.to_networkx()
        return nx_graph

    def networkx_to_dgl(self, nx_graph):
        dgl_graph = dgl.from_networkx(nx_graph)
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

        >>> # edge tensor 정보 출력
        >>> print(edge_tensor.shape)
        >>> print(edge_tensor)
        """
        # networkx 그래프의 엣지 정보를 가져옵니다
        edge_list = list(nx_graph.edges())
        num_edges = len(edge_list)

        # 엣지 리스트를 edge tensor로 변환합니다
        edge_tensor = np.zeros((2, num_edges), dtype=np.int32)
        # convert edge_tensor to tensor.long
        edge_tensor = torch.from_numpy(edge_tensor).long()

        for i in range(num_edges):
            edge_tensor[0][i] = edge_list[i][0]
            edge_tensor[1][i] = edge_list[i][1]

        return edge_tensor

    def edge_tensor_to_networkx(self, edge_tensor, num_nodes=0):
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

        >>> # 그래프 정보 출력
        >>> print("Nodes:", graph.nodes)
        >>> print("Edges:", graph.edges)
        """
        # edge_tensor의 shape과 데이터를 확인합니다
        num_edges = edge_tensor.shape[1]
        source_nodes = edge_tensor[0]
        dest_nodes = edge_tensor[1]

        # networkx 그래프 객체를 생성합니다
        graph = nx.Graph()

        if num_nodes != 0:
            graph.add_nodes_from(list(range(num_nodes)))

        # 각 엣지를 그래프에 추가합니다
        for i in range(num_edges):
            source_node = int(source_nodes[i].numpy())
            dest_node = int(dest_nodes[i].numpy())
            graph.add_edge(source_node, dest_node, weight=1)

        return graph
