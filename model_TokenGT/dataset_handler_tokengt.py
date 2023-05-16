import torch
import numpy as np


class TokenGTDataset(torch.utils.data.Dataset):
    def __init__(self, x, data: dict, device):
        """
        Args:
            x (tensor): the node features with dimension [#nodes, dim of a node feature]
            data (dict): the edge informations with keys ['edge_index_list', 'pedges', 'nedges']
        """
        super().__init__()
        self.x = x
        self.data = data
        self.device = device
        self.sample_num = 50  # sample number of subedges at once

    def sample_subedges(self, cur_edges, cur_edge_data):
        """
        Args:
            cur_edges (tensor): the edge informations with dimension [2 , #edges]
        """
        pool_edges = np.arange(cur_edges.shape[1])

        while pool_edges.shape[0] >= self.sample_num:
            subedges = np.random.choice(pool_edges, self.sample_num, replace=False)
            # delete the selected subedges from pool_edges
            pool_edges = np.setdiff1d(pool_edges, subedges)
            yield cur_edges[:, subedges], cur_edge_data[subedges]

        # yield the remaining subedges
        yield cur_edges[:, pool_edges], cur_edge_data[pool_edges]

    def convert_to_tokengt_input(self, time_t):
        cur_pedges = self.data["pedges"][time_t].long()
        cur_nedges = self.data["nedges"][time_t].long()
        cur_edges = torch.cat([cur_pedges, cur_nedges], dim=1)
        cur_x = self.x[time_t]
        cur_edge_data = torch.ones(cur_edges.shape[1], cur_x.shape[1])
        # select x from edges
        ## cur_x_for_edges = cur_x[cur_edges.unique()]

        # generate subgraphs
        node_num = []
        edge_num = []
        edge_index = []
        node_data = []
        edge_data = []

        # no batch_size version
        for subedges, subedge_data in self.sample_subedges(cur_edges, cur_edge_data):
            indices_subnodes = subedges.unique()
            mapping_dict = {
                int(value.numpy()): index
                for index, value in enumerate(indices_subnodes)
            }
            """
            Example: 
                original_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                indices = [2, 5, 8] 
                mapping_dict = {2: 0, 5: 1, 8: 2}
            """
            # select node features
            node_data += [cur_x[indices_subnodes]]
            edge_data += [subedge_data]
            # build numbers of nodes and edges
            node_num += [indices_subnodes.shape[0]]
            edge_num += [subedges.shape[1]]
            # convert each whole graph node index in subedges into the subgraph node index
            edge_index += [subedges.apply_(lambda x: mapping_dict[x])]

        node_data = torch.concat(node_data, dim=0).to(self.device)
        edge_data = torch.concat(edge_data, dim=0).to(self.device)
        edge_index = torch.concat(edge_index, dim=1).to(self.device)

        a_graph_at_a_time = {
            "node_data": node_data,
            "edge_data": edge_data,
            "edge_index": edge_index,
            "node_num": node_num,
            "edge_num": edge_num,
        }
        return a_graph_at_a_time  # represent one example in torch.utils.data.DataLoader

    def get_dim_of_x(self):
        return self.x.shape[1]

    def __getitem__(self, time_t):
        return self.convert_to_tokengt_input(time_t)

    def __len__(self):
        num_time_stamps = len(self.data["pedges"])
        return num_time_stamps


def test(args, x, data):
    dataset = TokenGTDataset(x, data, args.device)
    for index in range(len(dataset)):
        batch = dataset[index]
        print(batch)
