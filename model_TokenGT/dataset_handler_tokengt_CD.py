# enhanced version of dataset_handler_tokengt.py
# handle dataset with community detection to remove dupliacted nodes among subgraphs

from typing import Any
import torch
import numpy as np
from collections import defaultdict
from community import community_louvain
from community_dectection import CommunityDetection
from model_TokenGT.dataset_handler_tokengt import TokenGTDataset


class DatasetConverterCD:
    def __init__(self, node_data, mapping_dict):
        """
        Args:
            x (tensor): node_data after tokengt. the node features with dimension [#sum of nodes of graphs, dim of a node feature]
            mapping_dict (dict): the mapping from cur_x to node_data
            max_node_idx (int): the maximum node index of the original graph

        """
        self.node_data = node_data
        self.device = node_data.device
        self.mapping_dict = mapping_dict

    def __getitem__(self, indices):
        if isinstance(indices, torch.Tensor):
            # [0] is needed because mapping_dict[i] return list with one element
            indices = indices.cpu().apply_(lambda x: self.mapping_dict[x][0])
        return self.node_data[indices]

    def detach(self):
        self.node_data = self.node_data.detach()
        return self

    def new_ones(self, size):
        return self.node_data.new_ones(size)

    def new_zeros(self, size):
        return self.node_data.new_zeros(size)

    def to(self, device):
        self.node_data = self.node_data.to(device)
        return self


class TokenGTDatasetCD(TokenGTDataset):
    def __init__(self, x, data: dict, args):
        """
        이 클래스 자체가 dataset 역할을 수행함

        Args:
            x (tensor): the node features with dimension [#nodes, dim of a node feature]
            data (dict): the edge informations with keys ['edge_index_list', 'pedges', 'nedges']
        """
        super().__init__(x, data, args)

    def get_edges_from_node_indices(self, cur_edges, cur_edge_data, node_indices):
        bool_tensor = None
        for nodeid in node_indices:
            try:
                bool_tensor |= cur_edges == nodeid
            except:
                bool_tensor = cur_edges == nodeid
        # partition 내 node 간 edge 만 추출해야 하므로, all
        bool_tensor = bool_tensor.all(dim=0)

        return cur_edges[:, bool_tensor], cur_edge_data[bool_tensor]

    def generate_subgraphs_from_cd(self, partition, cur_edges, cur_edge_data, cur_x):
        # generate subgraphs
        node_num = []
        edge_num = []
        edge_index = []
        node_data = []
        edge_data = []

        self.mapping_dict_for_subgraphs_nodeid_to_original_nodeid = defaultdict(list)
        self.mapping_dict_for_original_nodeid_to_subgraphs_nodeid = defaultdict(list)
        total_indices_subnodes_with_edges = []
        # no batch_size version
        node_data_index = 0
        # convert partition to {partition_id: [nodeid1, nodeid2, ...]}
        new_partition = defaultdict(list)
        for nodeid, partition_id in partition.items():
            new_partition[partition_id] += [nodeid]

        for partition_id, indices_subnodes in new_partition.items():
            subedges, subedge_data = self.get_edges_from_node_indices(
                cur_edges, cur_edge_data, indices_subnodes
            )
            mapping_dict_for_edges_in_a_subgraph = {
                int(value): index for index, value in enumerate(indices_subnodes)
            }

            for cur_x_index in indices_subnodes:
                cur_x_index = int(cur_x_index)
                self.mapping_dict_for_subgraphs_nodeid_to_original_nodeid[
                    node_data_index
                ] += [cur_x_index]
                self.mapping_dict_for_original_nodeid_to_subgraphs_nodeid[
                    cur_x_index
                ] += [node_data_index]
                node_data_index += 1

            # select node features
            node_data += [cur_x[indices_subnodes]]
            edge_data += [subedge_data]
            # build numbers of nodes and edges
            node_num += [len(indices_subnodes)]
            edge_num += [subedges.shape[1]]
            # convert each whole graph node index in subedges into the subgraph node index
            edge_index += [
                subedges.apply_(lambda x: mapping_dict_for_edges_in_a_subgraph[x])
            ]
            total_indices_subnodes_with_edges += indices_subnodes

        node_data = torch.concat(node_data, dim=0).to(self.device)
        edge_data = torch.concat(edge_data, dim=0).to(self.device)
        edge_index = torch.concat(edge_index, dim=1).to(self.device)

        a_graph_at_a_time = {
            "node_data": node_data,
            "edge_data": edge_data,
            "edge_index": edge_index,
            "node_num": node_num,
            "edge_num": edge_num,
            "mapping_from_orig_to_subgraphs": self.mapping_dict_for_original_nodeid_to_subgraphs_nodeid,
            "mapping_from_subgraphs_to_orig": self.mapping_dict_for_subgraphs_nodeid_to_original_nodeid,
        }

        return (
            a_graph_at_a_time,
            total_indices_subnodes_with_edges,
        )  # represent one example in torch.utils.data.DataLoader

    def convert_to_tokengt_input(self, time_t):
        """
        (Pdb) p cur_pedges.shape
            torch.Size([2, 2624]) # [2, #pedges]
        (Pdb) p cur_x.shape
            torch.Size([23035, 32]) # [#nodes, dim of a node feature]
        """
        # TODO ANKI [OBNOTE: ] -  partition idea 정리
        # TODO idea: train edge 들만 가지고 partition 을 만들어야 정확하다고 생각됨
        # TODO 반대 idea: 어차피 t+1 때의 edge 를 맞추는 것이기 때문에, t 때의 edge 들로 partition 을 만들어도 상관없음
        # TODO END ANKI
        cur_edges = self.remove_duplicated_edges(self.data["pedges"][time_t].long())
        cur_x = self.x[time_t]
        cur_edge_data = torch.ones(cur_edges.shape[1], cur_x.shape[1])
        cd = CommunityDetection(self.args, cur_edges)
        partition = cd.partition

        # generate subgraphs
        (
            a_graph_at_a_time,
            total_indices_subnodes_with_edges,
        ) = self.generate_subgraphs_from_cd(partition, cur_edges, cur_edge_data, cur_x)

        a_graph_at_a_time_with_no_edge = self.generate_subgraphs_with_no_edges(
            cur_x, total_indices_subnodes_with_edges
        )

        # integrate two dicts
        a_graph_at_a_time["node_data"] = torch.concat(
            [
                a_graph_at_a_time["node_data"],
                a_graph_at_a_time_with_no_edge["node_data"],
            ],
            dim=0,
        )
        a_graph_at_a_time["node_num"] = (
            a_graph_at_a_time["node_num"] + a_graph_at_a_time_with_no_edge["node_num"]
        )
        a_graph_at_a_time["edge_num"] = (
            a_graph_at_a_time["edge_num"] + a_graph_at_a_time_with_no_edge["edge_num"]
        )

        return a_graph_at_a_time

    def get_dim_of_x(self):
        return self.x.shape[1]

    def __getitem__(self, time_t):
        # complete checking shuffled data
        # print(
        #    f'edge means: {self.converted_data_list[time_t]["edge_index"].to(float).mean().cpu().numpy()}'
        # )
        return self.converted_data_list[time_t]

    def __len__(self):
        num_time_stamps = len(self.data["pedges"])
        return num_time_stamps


def test(args, x, data):
    dataset = TokenGTDatasetCD(x, data, args.device)
    for index in range(len(dataset)):
        batch = dataset[index]
        print(batch)
