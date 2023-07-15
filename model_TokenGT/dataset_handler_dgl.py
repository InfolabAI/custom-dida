import torch


class OurDataset(torch.utils.data.Dataset):
    def __init__(self, x, args):
        """
        Args:
            x (tensor): the node features with dimension [#nodes, dim of a node feature]
            data (dict): the edge informations with keys ['edge_index_list', 'pedges', 'nedges']
        """
        self.x = x
        self.args = args
        self.device = args.device
        self.sample_num_node_with_no_edge = 100
