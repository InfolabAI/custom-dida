import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from community import community_louvain
import os


class CommunityDetection:
    def __init__(self, args, dataset, edge_tensor_list):
        if args.draw_community_detection == 0:
            return

        self.dataset = dataset
        self.save_path = f"logs/graphs/{dataset}"

        os.makedirs(self.save_path, exist_ok=True)

        for time_t, edge_tensor in enumerate(edge_tensor_list):
            self.time_t = time_t
            G = self.edge_tensor_to_graph(edge_tensor)
            partition = community_louvain.best_partition(G)
            self.draw_nx_graph(G, partition, nx.spring_layout, k=0.05)

        exit(0)

    def test(self, edge_tensor):
        G = self.edge_tensor_to_graph(edge_tensor)
        partition = community_louvain.best_partition(G)
        # TODO ANKI [OBNOTE: ] - k means the optimal distance between nodes
        # draw the graph
        # example link: https://frhyme.github.io/python-lib/networkx_layout/
        self.draw_nx_graph(
            G, partition, nx.spring_layout, k=0.05, scale=0.1, iterations=50
        )
        # self.draw_nx_graph(
        #    G, partition, nx.spring_layout, k=0.01, scale=0.1, iterations=50
        # )
        # self.draw_nx_graph(G, partition, nx.spring_layout, k=0.1, scale=1)
        # self.draw_nx_graph(G, partition, nx.spring_layout, k=0.1, scale=10)
        # TODO END ANKI
        # self.draw_nx_graph(G, partition, nx.spectral_layout)
        # self.draw_nx_graph(G, partition, nx.circular_layout)
        # self.draw_nx_graph(G, partition, nx.kamada_kawai_layout)
        # self.draw_nx_graph(G, partition, nx.random_layout)
        # self.draw_nx_graph(G, partition, nx.shell_layout)
        # self.draw_nx_graph(G, partition, nx.spiral_layout)

    @classmethod
    def edge_tensor_to_graph(self, edge_tensor):
        """
        Examples:
            # 주어진 edge tensor
            edge_tensor = np.array([[23034, 23034, 1605, 11045, 11045, 11045],
                                [1605, 4350, 23034, 11048, 11047, 11046]])

            # edge tensor를 그래프로 변환
            graph = edge_tensor_to_graph(edge_tensor)

            # 그래프 정보 출력
            print("Nodes:", graph.nodes)
            print("Edges:", graph.edges)
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

    # TODO ANKI [OBNOTE: ] - How to send kwargs to function
    def draw_nx_graph(self, G, partition, layoutF=nx.spring_layout, **kwargs):
        """
        Args:
            G: networkx graph
            partition: dict of node and its partition
        """
        pos = layoutF(G, **kwargs)
        # TODO END ANKI
        # color the nodes according to their partition
        cmap = cm.get_cmap("cool", max(partition.values()) + 1)

        im = nx.draw_networkx_nodes(
            G,
            pos,
            partition.keys(),
            node_size=1,
            cmap=cmap,
            node_color=list(partition.values()),
        )
        nx.draw_networkx_edges(G, pos, alpha=1, width=0.1)
        # TODO ANKI [OBNOTE: ] - Labels means texts to be displayed on nodes
        nx.draw_networkx_labels(
            G, pos, font_size=1, labels=partition, font_color="black"
        )
        # TODO END ANKI
        plt.colorbar(im)
        # TODO ANKI [OBNOTE: ] - convert function into text
        plt.savefig(f"{self.save_path}/{self.time_t}.png", dpi=2000)
        # TODO END ANKI
        plt.cla()
        plt.clf()
