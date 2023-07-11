# In this version, there are a lot of duplicated nodes among subgraphs, so, forward and backward propagation are slow
from typing import Any
import torch
import numpy as np
from collections import defaultdict
from model_TokenGT.dataset_handler_tokengt import TokenGTDataset
from util_hee import remove_duplicated_edges


class DatasetConverter_noCD:
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
        original_node_indices = np.array(list(self.mapping_dict.keys()))
        original_node_indices.sort()
        self.max_node_idx = original_node_indices[-1]
        self.node_data = self.convert_to_original()

    def getitem_an_index(self, idx):
        gathered_x = self.node_data[self.mapping_dict[idx]]
        return gathered_x.sum(0, keepdim=True)

    def convert_to_original(self):
        tensor_list = []
        for i in range(self.max_node_idx + 1):
            tensor_list += [self.getitem_an_index(i)]
        return torch.cat(tensor_list, dim=0)

    def __getitem__(self, idx):
        return self.node_data[idx]

    # def __getitem__(self, idx):
    #    if isinstance(idx, int):
    #        self.getitem_an_index(idx)
    #    else:
    #        tensor_list = []
    #        for el in idx:
    #            if isinstance(el, int):  # 이미 int인 경우 그대로 반환
    #                pass
    #            elif isinstance(el, np.ndarray):  # NumPy 배열인 경우 int로 변환하여 반환
    #                el = int(el)
    #            elif isinstance(el, torch.Tensor):  # PyTorch Tensor인 경우 int로 변환하여 반환
    #                el = int(el.cpu().numpy())
    #            else:  # 지원하지 않는 타입인 경우 예외 처리
    #                raise NotImplementedError

    #            tensor_list += [self.getitem_an_index(el)]
    #        return torch.concat(tensor_list, dim=0)

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


class TokenGTDataset_noCD(TokenGTDataset):
    def __init__(self, x, data: dict, args):
        self.sample_num_edge = 50  # sample number of subedges at once
        super().__init__(x, data, args)

    def sample_subedges(self, cur_edges, cur_edge_data):
        """
        Args:
            cur_edges (tensor): the edge informations with dimension [2 , #edges]
        """
        pool_edges = np.arange(cur_edges.shape[1])

        while pool_edges.shape[0] >= self.sample_num_edge:
            subedges = np.random.choice(pool_edges, self.sample_num_edge, replace=False)
            # delete the selected subedges from pool_edges
            pool_edges = np.setdiff1d(pool_edges, subedges)
            yield cur_edges[:, subedges], cur_edge_data[subedges]

        # TODO ANKI [OBNOTE: ] - # pool_edges 의 길이가 sameple_num_edge 로 정확히 나누어 떨어지면 pool_edges 길이가 0일 수 있으므로 예외처리
        if pool_edges.shape[0] != 0:
            # pool_edges 의 길이가 sameple_num_edge 로 정확히 나누어 떨어지면 pool_edges 길이가 0일 수 있으므로 예외처리
            # yield the remaining subedges
            yield cur_edges[:, pool_edges], cur_edge_data[pool_edges]
            # TODO END ANKI

    def generate_subgraphs_with_edges(self, cur_edges, cur_edge_data, cur_x):
        # generate subgraphs
        node_num = []
        edge_num = []
        edge_index = []
        node_data = []
        edge_data = []

        self.mapping_dict_for_original_nodeid_to_subgraphs_nodeid = defaultdict(list)
        self.mapping_dict_for_subgraphs_nodeid_to_original_nodeid = defaultdict(list)
        total_indices_subnodes_with_edges = []
        # no batch_size version
        node_data_index = 0
        for subedges, subedge_data in self.sample_subedges(cur_edges, cur_edge_data):
            indices_subnodes = subedges.unique().tolist()
            mapping_dict_for_edges_in_a_subgraph = {
                int(value): index for index, value in enumerate(indices_subnodes)
            }
            """
            Example: 
                original_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                indices = [2, 5, 8] 
                mapping_dict = {2: 0, 5: 1, 8: 2}
            """
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
            # TODO ANKI [OBNOTE: ] - torch.tensor.apply_ is the function like map() in Python
            edge_index += [
                subedges.apply_(lambda x: mapping_dict_for_edges_in_a_subgraph[x])
            ]
            # TODO END ANKI
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
        """
        (Pdb) p a_graph_at_a_time['node_data'].shape
            torch.Size([9923, 32]) # [#sum of nodes of graphs, dim of a node feature]
        (Pdb) p a_graph_at_a_time['edge_data'].shape
            torch.Size([5248, 32]) # [#sum of edges of graphs, dim of a node feature]
        (Pdb) p a_graph_at_a_time['edge_index'].shape
            torch.Size([2, 5248]) # [2, #sum of edges of graphs]
        (Pdb) p len(a_graph_at_a_time['node_num'])
            105 # number of graphs
        (Pdb) p len(a_graph_at_a_time['edge_num'])
            105 # number of graphs 
        """

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
        cur_edges = self.data["pedges"][time_t].long()
        cur_edges = remove_duplicated_edges(cur_edges)
        cur_x = self.x[time_t]
        cur_edge_data = torch.ones(cur_edges.shape[1], cur_x.shape[1])
        # select x from edges
        ## cur_x_for_edges = cur_x[cur_edges.unique()]

        # generate subgraphs
        (
            a_graph_at_a_time,
            total_indices_subnodes_with_edges,
        ) = self.generate_subgraphs_with_edges(cur_edges, cur_edge_data, cur_x)

        a_graph_at_a_time_with_no_edge = self.generate_subgraphs_with_no_edges(
            cur_x, total_indices_subnodes_with_edges
        )

        if a_graph_at_a_time_with_no_edge is None:
            return a_graph_at_a_time

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

    def extract_batch_from_a_graph_at_a_time(self, a_graph_at_a_time):
        # This cannot be used because embeddings for all the nodes at time t are required to calculate the loss
        pass
        # batch_graph = {}
        # offset_node = 0
        # offset_edge = 0
        # for i in range(0, len(a_graph_at_a_time["node_num"]), self.batch_size):
        #    indices = np.arange(i, i + self.batch_size)
        #    batch_graph["node_num"] = np.array(a_graph_at_a_time["node_num"])[
        #        indices
        #    ].tolist()
        #    batch_graph["edge_num"] = np.array(a_graph_at_a_time["edge_num"])[
        #        indices
        #    ].tolist()

        #    batch_graph["node_data"] = a_graph_at_a_time["node_data"][
        #        offset_node : sum(batch_graph["node_num"])
        #    ]
        #    offset_node += sum(batch_graph["node_num"])
        #    batch_graph["edge_data"] = a_graph_at_a_time["edge_data"][
        #        offset_edge : sum(batch_graph["edge_num"])
        #    ]
        #    batch_graph["edge_index"] = a_graph_at_a_time["edge_index"][
        #        :, offset_edge : sum(batch_graph["edge_num"])
        #    ]
        #    offset_edge += sum(batch_graph["edge_num"])
        #    yield batch_graph

    def reverse_cur_x(self):
        """
        Return cur_x extracted from a_graph_at_a_time['node_data']
        """

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
    dataset = TokenGTDataset_noCD(x, data, args.device)
    for index in range(len(dataset)):
        batch = dataset[index]
        print(batch)
