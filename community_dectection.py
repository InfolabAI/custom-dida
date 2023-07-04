from community import community_louvain
from convert_graph_types import ConvertGraphTypes
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
        self.args = args
        cgt = ConvertGraphTypes()
        G = cgt.edge_tensor_to_networkx(edge_tensor)
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
        if self.args.dataset == "collab":
            # To save memory, use resolution=0.01, but, reduced memory is not enough (23.5G -> 17.5G)
            # partition = community_louvain.best_partition(G, resolution=0.01)
            partition = community_louvain.best_partition(G)
        else:
            partition = community_louvain.best_partition(G)
        return partition
