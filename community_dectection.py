import torch
import numpy as np
import networkx as nx
from community import community_louvain
from convert_graph_types import ConvertGraphTypes


class CommunityDetection:
    def __init__(self, args, dglG, input_partition=None):
        """
        Parameters
        ----------
        args : argparse
        edge_tensor : [2, num_edges]
        """
        self.args = args
        input_partition = self.sync_partition_with_dglG(dglG, input_partition)
        if input_partition is not None:
            self.partition = input_partition
            return

        cgt = ConvertGraphTypes()
        G = cgt.dgl_to_networkx(dglG.cpu())

        # CD 속도 향상을 위해 edge 없는 node 정보 제거
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)

        self.version = args.model
        if self.version == "tokengt_cd" or self.version == "ours":
            partition = self.louvain(G, input_partition)
        elif self.version == "tokengt_cdrandom":
            partition = self.random(G)
            pass
        elif self.version == "tokengt_nocd":
            partition = None
        else:
            raise NotImplementedError("Community detection version is not implemented.")
        self.G = G
        self.partition = partition

    def sync_partition_with_dglG(self, dglG, input_partition):
        # sync node indices between G and input_partition
        if input_partition is not None:
            dglG_tensor = dglG.adj().indices().flatten().unique()
            part_tensor = torch.tensor(list(input_partition.keys())).to(
                dglG_tensor.device
            )
            # get indices in G that are not in input_partition
            indices = dglG_tensor[torch.isin(dglG_tensor, part_tensor, invert=True)]
            # partition 0 부터 차례대로 하나씩 넣어줌
            for i in range(len(indices)):
                input_partition[int(indices[i].cpu())] = i

        return input_partition

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

    def louvain(self, G, input_partition):
        """
        Parameters
        ----------
        G : networkx graph
        """
        if self.args.dataset == "collab":
            # To save memory, use resolution=0.01, but, reduced memory is not enough (23.5G -> 17.5G)
            # partition = community_louvain.best_partition(G, resolution=0.01)
            partition = community_louvain.best_partition(G, partition=input_partition)
        else:
            partition = community_louvain.best_partition(G, partition=input_partition)
        return partition
