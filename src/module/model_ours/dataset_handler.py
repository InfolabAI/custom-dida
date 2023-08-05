# handle subgraph dataset
import torch


class DatasetConverter_CD:
    def __init__(self, node_data, mapping_dict):
        """
        Args:
            x (tensor): node_data after tokengt. the node features with dimension [#sum of nodes of graphs, dim of a node feature]
            mapping_dict (dict): the mapping from original node id to node id in subgraphs

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
