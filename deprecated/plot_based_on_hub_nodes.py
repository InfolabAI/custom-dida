import torch
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from deprecated.plot_parent import Plot
from loguru import logger


class PlotBasedOnHubNodes(Plot):
    def __init__(self, args, data, decoder, writer, epoch):
        super().__init__(args, data, decoder, writer, epoch)
        # {degree: [accuracy1, accuracy2, ...(for each time step)]}
        self.degree_acc = defaultdict(list)

    def process_t(self, z, time_t):
        if self.args.plot_hub_nodes == 0:
            return

        logger.info(f"plot time {time_t}")
        edges = self.data["pedges"][time_t].long().to(self.args.device)
        graph = self.get_graph_dict(self.data["pedges"][time_t])
        node_degree = self.get_node_degree_dict(self.data["pedges"][time_t])
        self.plot_node_degree_histogram(time_t, edges, self.epoch)
        self.gather_node_degree_acc(z, edges, node_degree)

    def process_epoch(self, epoch):
        if self.args.plot_hub_nodes == 0:
            return

        logger.info(f"plot epoch {epoch}")
        self.plot_node_degree_acc_scalar(epoch)

    def get_graph_dict(self, edges: torch.Tensor) -> dict:
        """
        Return graph that consists of only outlinks of each node
        Args:
            edges: shape [2, num_edges]
        Returns:
            graph: dict {nodeid: [outlink1, outlink2, ...]}
        """
        assert isinstance(edges, torch.Tensor)

        graph = defaultdict(list)
        for i in range(edges.shape[1]):
            start_node, end_node = int(edges[0, i].numpy()), int(edges[1, i])
            graph[start_node].append(end_node)

        graph = dict(sorted(graph.items(), key=lambda x: len(x[1]), reverse=True))
        return graph

    def get_node_degree_dict(self, edges: torch.Tensor) -> dict:
        """
        Args:
            edges: shape [2, num_edges]
        Returns:
            node_degree: dict {degreeN: [node1_of_degreeN, node2_of_degreeN, ...]}
        """
        assert isinstance(edges, torch.Tensor)

        node_degree = defaultdict(list)
        unique, degree = edges.unique(return_counts=True)
        # sorted_indices = torch.argsort(degree, descending=True)
        for i in range(len(unique)):
            node_degree[int(degree[i].numpy())].append(int(unique[i].numpy()))

        node_degree = dict(
            sorted(node_degree.items(), key=lambda x: x[0], reverse=True)
        )
        return node_degree

    def plot_node_degree_histogram(self, time_t, edges, epoch):
        # only plot for the first epoch
        if epoch != 1:
            return

        # plot per time step
        _, d = edges.unique(return_counts=True)
        self.writer.add_histogram(f"node_degree_histogram", d, global_step=time_t)

    def gather_node_degree_acc(self, z, edges, node_degree):
        # only plot for the last time step per a epoch
        pos_y = torch.ones(edges.size(1)).numpy().astype(int)
        pos_pred = (self.decoder(z, edges).detach().cpu().numpy() >= 0.5).astype(int)
        edges = edges.cpu()

        degree_edge_index = torch.zeros(edges.size(1), dtype=torch.bool)
        for degree, nodes in node_degree.items():
            degree_edge_index.fill_(0)
            for node in nodes:
                degree_edge_index_src = edges[0, :] == node
                degree_edge_index_dst = edges[1, :] == node
                degree_edge_index |= degree_edge_index_src | degree_edge_index_dst

            self.degree_acc[degree] += [
                accuracy_score(pos_y[degree_edge_index], pos_pred[degree_edge_index])
            ]

    def plot_node_degree_acc_scalar(self, epoch):
        self.degree_acc = dict(
            sorted(self.degree_acc.items(), key=lambda x: x[0], reverse=True)
        )
        for degree, acc in self.degree_acc.items():
            acc = np.array(acc).mean()
            self.writer.add_scalar(
                f"degree_acc for epoch {epoch}", acc, global_step=degree
            )

    def test(self):
        self.get_graph_dict("hi")
