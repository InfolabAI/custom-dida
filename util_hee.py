import torch
from datetime import datetime


def get_current_datetime():
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d %H:%M")
    return formatted_date


def remove_duplicated_edges(edges):
    """
    Args:
        edges (tensor): [2, #edges]
    """
    # duplicated edges are [src1, dst1] and [dst1, src1], so we sort them and remove the duplicated ones
    # TODO ANKI [OBNOTE: ] - For example, a duplicated edge is [src1, dst1] and [dst1, src1], so we sort them and remove the duplicated ones
    # edges (tensor): [2, #edges]
    original_edge_num = edges.shape[1]
    edges = (torch.sort(edges, dim=0)[0]).unique(dim=1)
    remove_dup_edge_num = edges.shape[1]
    edges = edges[:, edges[0] != edges[1]]
    remove_self_loop = edges.shape[1]
    print(
        f"Remove duplicated edges: {original_edge_num - remove_dup_edge_num} and self-loop edges: {remove_dup_edge_num - remove_self_loop}"
    )
    return edges

    # TODO END ANKI
