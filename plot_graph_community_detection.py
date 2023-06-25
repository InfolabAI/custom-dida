import torch
import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from community_dectection import CommunityDetection
from plot_sparsity_adjmat import plot_sparsity_pattern
from community import community_louvain
from collections import defaultdict
import os


class PlotGraphMat:
    def __init__(self, args, dataset, edge_tensor_list):
        """
        오직 plot 을 위한 class. 이 class 없이도 best_partition 을 수행하는 함수 따로 있으니 검색할 것.
        """
        assert (
            args.plot_graphs_community_detection != 1 or args.plot_sparsity_mat_cd != 1
        )

        self.dataset = dataset

        for time_t, edge_tensor in enumerate(edge_tensor_list):
            self.time_t = time_t
            cd = CommunityDetection(args, edge_tensor)

            if args.plot_graphs_community_detection == 1:
                self.save_path = f"logs/graphs/{args.model}_{dataset}"
                os.makedirs(self.save_path, exist_ok=True)
                self.plot_nx_graph(cd.G, cd.partition, nx.spring_layout, k=0.05)
            elif args.plot_sparsity_mat_cd == 1:
                self.save_path = f"logs/sparsity_mat/{args.model}_{dataset}"
                # partition 번호: node 번호 로 변환
                partition_to_node_dict = defaultdict(list)
                for nodei, partitioni in cd.partition.items():
                    partition_to_node_dict[partitioni].append(nodei)
                # 각 community 마다의 node list 를 순서대로 입력 후, 1차원 tensor 로 변환
                nodes_in_order_of_communities = []
                for nodes_partitioni in partition_to_node_dict.values():
                    nodes_in_order_of_communities.append(torch.tensor(nodes_partitioni))
                nodes_in_order_of_communities = torch.cat(nodes_in_order_of_communities)
                # node id 를 community 순서대로의 id 로 변환. 예를 들어, [20345, 592, 213] 가 community 1 이면, edge_tensor 에서 20345->0, 592->1, 213->2 로 변환
                nodei_communitynodei = {}
                for i, nodei in enumerate(nodes_in_order_of_communities):
                    nodei_communitynodei[int(nodei.numpy())] = i
                edge_tensor.apply_(lambda x: nodei_communitynodei[x])
                # 변환한 edge tensor 로 G 다시 만들고 sparsity mat plot
                cd = CommunityDetection(args, edge_tensor)
                os.makedirs(self.save_path, exist_ok=True)
                self.plot_sparsity_mat(cd.G)
            else:
                return

        exit(0)

    def test(self, edge_tensor):
        cd = CommunityDetection(edge_tensor)
        # TODO ANKI [OBNOTE: ] - k means the optimal distance between nodes
        # draw the graph
        # example link: https://frhyme.github.io/python-lib/networkx_layout/
        self.plot_nx_graph(
            cd.G, cd.partition, nx.spring_layout, k=0.05, scale=0.1, iterations=50
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

    # TODO ANKI [OBNOTE: ] - How to send kwargs to function
    def plot_nx_graph(self, G, partition, layoutF=nx.spring_layout, **kwargs):
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

    def plot_sparsity_mat(self, G):
        """
        Parameters
        ----------
        G: networkx graph
        partition: dict of node and its partition
        """

        # trouble shooting
        edges_dict = defaultdict(list)
        for i, j in G.edges:
            edges_dict[i].append(j)

        adj = nx.adjacency_matrix(G, nodelist=list(range(0, len(G.nodes())))).toarray()
        plot_sparsity_pattern(adj, f"{self.save_path}/{self.time_t}.png")
        plt.cla()
        plt.clf()
