import torch
import numpy as np


class TokenGTDataset(torch.utils.data.Dataset):
    def __init__(self, x, data: dict, args):
        """
        Args:
            x (tensor): the node features with dimension [#nodes, dim of a node feature]
            data (dict): the edge informations with keys ['edge_index_list', 'pedges', 'nedges']
        """
        self.x = x
        self.data = data
        self.args = args
        self.device = args.device
        self.max_time = len(self.data["pedges"])
        self.sample_num_node_with_no_edge = 100
        self.convert_all()

    def convert_all(self):
        self.converted_data_list = []
        for t in range(self.max_time):
            self.converted_data_list += [self.convert_to_tokengt_input(t)]

    def remove_duplicated_edges(self, edges):
        """
        Args:
            edges (tensor): [2, #edges]
        """
        # duplicated edges are [src1, dst1] and [dst1, src1], so we sort them and remove the duplicated ones
        # TODO ANKI [OBNOTE: ] - For example, a duplicated edge is [src1, dst1] and [dst1, src1], so we sort them and remove the duplicated ones
        # edges (tensor): [2, #edges]
        return (torch.sort(edges, dim=0)[0]).unique(dim=1)
        # TODO END ANKI

    # TODO ANKI [OBNOTE: ] -
    def generate_subgraphs_with_no_edges(
        self, cur_x, total_indices_subnodes_with_edges
    ):
        # To caluculate the loss, we need all the nodes including nodes with no edges.
        # TODO END ANKI

        indices_nodes_with_no_edge = np.setdiff1d(
            np.arange(cur_x.shape[0]), total_indices_subnodes_with_edges
        )
        node_num = []
        edge_num = []
        # edge_index = []
        node_data = []
        # edge_data = []
        node_data_index = len(self.mapping_dict_for_subgraphs_nodeid_to_original_nodeid)
        for idx in range(
            0, len(indices_nodes_with_no_edge), self.sample_num_node_with_no_edge
        ):
            indices_for_a_subgraph = indices_nodes_with_no_edge[
                idx : idx + self.sample_num_node_with_no_edge
            ]
            for cur_x_index in indices_for_a_subgraph:
                cur_x_index = int(cur_x_index)
                self.mapping_dict_for_subgraphs_nodeid_to_original_nodeid[
                    node_data_index
                ] += [cur_x_index]
                self.mapping_dict_for_original_nodeid_to_subgraphs_nodeid[
                    cur_x_index
                ] += [node_data_index]
                node_data_index += 1

            # select node features
            node_data += [cur_x[indices_for_a_subgraph]]
            # edge_data += [subedge_data]
            # build numbers of nodes and edges
            node_num += [len(indices_for_a_subgraph)]
            edge_num += [0]
            # convert each whole graph node index in subedges into the subgraph node index
            # edge_index += [subedges.apply_(lambda x: mapping_dict[x])]

        node_data = torch.concat(node_data, dim=0).to(self.device)

        a_graph_at_a_time = {
            "node_data": node_data,
            "node_num": node_num,
            "edge_num": edge_num,
        }
        return a_graph_at_a_time
