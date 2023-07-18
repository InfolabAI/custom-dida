import torch
import numpy as np
import networkx as nx
import math
from tqdm import tqdm
from collections import defaultdict
from loguru import logger
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
        self.dglG = dglG
        input_partition = self.sync_partition_with_dglG(dglG, input_partition)
        if input_partition is not None:
            self.partition = input_partition
            return

        cgt = ConvertGraphTypes()
        G = cgt.dgl_to_networkx(dglG.cpu())

        # CD 속도 향상을 위해 edge 없는 node 정보 제거
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
        self.G = G

    def get_merged_communities(self, num_comm_groups=1, input_partition=None):
        """
        원하는 수의 community group 을 생성합니다.
        Parameters
        ----------
        - num_comm_groups : int
            - community group 의 수

        Examples
        --------
        >>> comm_groups[i]
            {comm_id : [node_id_0, node_id_1, ...]}
        """
        comm_groups = []
        resolution = 1.0
        # resolution 을 바꾸어 각 comm_groups[i] 가 다른 comm_group 되게 함
        for i in tqdm(
            range(num_comm_groups), desc="generating community group", leave=False
        ):
            partition = self._get_raw_communities(input_partition, resolution)
            comm_group = self._merge_communities(
                partition, self.args.minnum_nodes_for_a_community
            )
            comm_group = self._merge_deactivated_node_communities(comm_group)
            comm_groups.append(comm_group)
            self._analyze_community_group(comm_group)
            resolution *= 0.2
        return comm_groups

    def _merge_deactivated_node_communities(self, activated_node_communities):
        """
        activated node communties 에 포함되지 않은 deactivated node 로 구성된 communities 를 activated_node_communities 에 merge
        """
        entire_nodes = np.arange(self.dglG.num_nodes())
        activated_nodes = torch.concat(self.dglG.edges()).unique().cpu().numpy()
        deactivated_nodes = np.setdiff1d(entire_nodes, activated_nodes)
        # randomly sort deactivated nodes
        np.random.shuffle(deactivated_nodes)

        # we assume that the number of nodes in each deactivated node community is the same as the number of edges in the community
        num_nodes_for_each_deactivated_node_community = (
            self.args.minnum_nodes_for_a_community * 2
        )

        # divide deactivated nodes into subsets based on num_nodes_for_each_deactivated_node_community
        comm_id = np.array(list(activated_node_communities.keys())).max() + 1
        for idx in range(
            0, len(deactivated_nodes), num_nodes_for_each_deactivated_node_community
        ):
            activated_node_communities[comm_id] = deactivated_nodes[
                idx : idx + num_nodes_for_each_deactivated_node_community
            ]
            comm_id += 1
        return activated_node_communities

    def _merge_communities(self, partition, minnum_nodes_for_a_community):
        """
        - partition 구조 {node id : comm id}  -convert->  community 구조 {comm id : [node_1, node_2, ...]
        - 동시에 각 community 의 node 수가 minnum_nodes_for_a_community 보다 크도록 community 를 merge 함. 너무 많은 community 가 생기는 것을 방지하기 위함.

        Parameters
        ----------
        - partition : dict
        - minnum_nodes_for_a_community : int
            - community의 최소 노드 수
        """
        # convert partition to {partition_id: [nodeid1, nodeid2, ...]}
        communities = defaultdict(list)
        for nodeid, partition_id in partition.items():
            communities[partition_id] += [nodeid]

        num_nodes_in_partition = torch.concat(self.dglG.edges()).unique().shape[0]
        num_communities = math.ceil(
            num_nodes_in_partition / minnum_nodes_for_a_community
        )

        # restrict the number of communities
        partition_bins = {}
        for i in range(num_communities):
            partition_bins[i] = []

        # 큰 community 부터 차례대로 partition_bins 통에 채워넣음. 모든 partition_bins 통이 거의 같은 #nodes 를 가지도록 하기 위함
        for com_id, node_list in dict(
            sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        ).items():
            minlen_idx = np.argmin(np.array([len(x) for x in partition_bins.values()]))
            partition_bins[minlen_idx] += node_list
        communities = partition_bins
        return communities

    def _get_raw_communities(self, input_partition=None, resolution=1.0):
        """
        activated node 로 구성된 partition 을 생성
        """
        if self.args.comm_alg == "louvain":
            partition = self._louvain(self.G, input_partition, resolution=resolution)
        elif self.args.comm_alg == "random":
            partition = self._random(self.G)
        else:
            raise NotImplementedError("Community detection version is not implemented.")
        return partition

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

    def _random(self, G):
        """
        Parameters
        ----------
        G : networkx graph
        """

        # get community size
        partition = self._louvain(G)
        community_size = max(partition.values()) + 1

        # randomize community
        for nodei, communityi in partition.items():
            partition[nodei] = np.random.randint(0, community_size)

        return partition

    def _louvain(self, G, input_partition, resolution=1.0):
        """
        Parameters
        ----------
        G : networkx graph
        """
        partition = community_louvain.best_partition(
            G, partition=input_partition, resolution=resolution
        )
        return partition

    def _analyze_community_group(self, comm_group):
        logger.debug(f"Comm number max {np.array(list(comm_group.keys())).max()}")
        logger.debug(f"Comm number mean {np.array(list(comm_group.keys())).mean()}")
        logger.debug(f"Comm number min {np.array(list(comm_group.keys())).min()}")
        logger.debug(
            f"Comm size max {np.array([len(v) for k, v in comm_group.items()]).max()}"
        )
        logger.debug(
            f"Comm size mean {np.array([len(v) for k, v in comm_group.items()]).mean()}"
        )
        logger.debug(
            f"Comm size min {np.array([len(v) for k, v in comm_group.items()]).min()}"
        )

    def _analyze_partition(self, partition):
        logger.debug(f"Comm number max {np.array(list(partition.values())).max()}")
        logger.debug(f"Comm number mean {np.array(list(partition.values())).mean()}")
        logger.debug(f"Comm number min {np.array(list(partition.values())).min()}")
