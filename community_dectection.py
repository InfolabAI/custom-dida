from community import community_louvain
import numpy as np
import networkx as nx


class CommunityDetection:
    def __init__(self, args, edge_tensor):
        """
        Parameters
        ----------
        args : argparse
        edge_tensor : [2, num_edges]
        """
        G = self.edge_tensor_to_graph(edge_tensor)
        self.version = args.model
        if self.version == "tokengt_cd":
            partition = self.louvain(G)
        elif self.version == "tokengt_cdrandom":
            partition = self.random(G)
            pass
        elif self.version == "tokengt_nocd":
            partition = None
        else:
            raise NotImplementedError("Community detection version is not implemented.")
        self.G = G
        self.partition = partition

    def random(self, G):
        """
        Parameters
        ----------
        G : networkx graph
        """
        # get community size
        partition = self.louvain(G)
        community_size = max(partition.values()) + 1

        # randomize community
        for nodei, communityi in partition.items():
            partition[nodei] = np.random.randint(0, community_size)

        return partition

    def louvain(self, G):
        """
        Parameters
        ----------
        G : networkx graph
        """
        partition = community_louvain.best_partition(G)
        return partition

    @classmethod
    def edge_tensor_to_graph(self, edge_tensor):
        """
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
        num_nodes, num_edges = edge_tensor.shape
        source_nodes = edge_tensor[0]
        dest_nodes = edge_tensor[1]

        # networkx 그래프 객체를 생성합니다
        graph = nx.Graph()

        # 각 엣지를 그래프에 추가합니다
        for i in range(num_edges):
            source_node = int(source_nodes[i].numpy())
            dest_node = int(dest_nodes[i].numpy())
            graph.add_edge(source_node, dest_node, weight=1)

        return graph
