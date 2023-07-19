import os
import torch
import random
import numpy as np
import torch
import subprocess
from loguru import logger
from datetime import datetime


def get_gpu_memory_usage(device_id):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    memory_usage = [int(x) for x in output.split("\n")]
    return memory_usage[int(device_id)]


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
    # edges (tensor): [2, #edges]
    original_edge_num = edges.shape[1]
    edges = (torch.sort(edges, dim=0)[0]).unique(dim=1)
    remove_dup_edge_num = edges.shape[1]
    edges = edges[:, edges[0] != edges[1]]
    remove_self_loop = edges.shape[1]
    logger.debug(
        f"Remove duplicated edges: {original_edge_num - remove_dup_edge_num} and self-loop edges: {remove_dup_edge_num - remove_self_loop}"
    )
    return edges


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_arg_dict(args):
    info_dict = args.__dict__
    ks = list(info_dict.keys())
    arg_dict = {}
    for k in ks:
        v = info_dict[k]
        for t in [int, float, str, bool, torch.Tensor]:
            if isinstance(v, t):
                arg_dict[k] = v
                break
    return arg_dict


def setargs(args, hp):
    for k, v in hp.items():
        setattr(args, k, v)


def merge_namespaces(namespace1, namespace2):
    """
    Update namespace1 with items from namespace2."""
    dict_namespace1 = vars(namespace1)
    dict_namespace2 = vars(namespace2)
    for key in set(dict_namespace1.keys()).intersection(set(dict_namespace2.keys())):
        logger.warning(
            f"While merging namespaces, for {key}, {dict_namespace1[key]} is replaced by {dict_namespace2[key]}"
        )
    dict_namespace1.update(dict_namespace2)
    setargs(namespace1, dict_namespace1)

    return namespace1


from gensim.models import Word2Vec
import gensim


def sen2vec(sentences, vector_size=32):
    """use gensim.word2vec to generate wordvecs, and average as sentence vectors.
    if exception happens use zero embedding.
    @ params sentences : list of sentence
    @ params vector_size
    @ return : sentence embedding
    """
    sentences = [list(gensim.utils.tokenize(a, lower=True)) for a in sentences]
    model = Word2Vec(sentences, vector_size=vector_size, min_count=1)
    logger.info("word2vec done")
    embs = []
    for s in sentences:
        try:
            emb = model.wv[s]
            emb = np.mean(emb, axis=0)
        except Exception as e:
            logger.info(e)
            emb = np.zeros(vector_size)
        embs.append(emb)
    embs = np.stack(embs)
    logger.info(f"emb shape : {embs.shape}")
    return embs


from torch_geometric.utils.negative_sampling import negative_sampling


def hard_negative_sampling(edges, all_neg=False, inplace=False):
    ei = edges.numpy()

    # reorder
    nodes = list(set(ei.flatten()))
    nodes.sort()
    id2n = nodes
    n2id = dict(zip(nodes, np.arange(len(nodes))))

    ei_ = np.vectorize(lambda x: n2id[x])(ei)

    if all_neg:
        maxn = len(nodes)
        nei_ = []
        pos_e = set([tuple(x) for x in ei_.T])
        for i in range(maxn):
            for j in range(maxn):
                if i != j and (i, j) not in pos_e:
                    nei_.append([i, j])
        nei_ = torch.LongTensor(nei_).T
    else:
        nei_ = negative_sampling(torch.LongTensor(ei_))
    nei = torch.LongTensor(np.vectorize(lambda x: id2n[x])(nei_.numpy()))
    return nei


def bi_negative_sampling(edges, num_nodes, shift):
    # shift forces that edges which has one of nodeid grater (XOR) than shift  to be source and dest nodes
    nes = edges.new_zeros((2, 0))
    while nes.shape[1] < edges.shape[1]:
        num_need = edges.shape[1] - nes.shape[1]
        ne = negative_sampling(edges, num_nodes=num_nodes, num_neg_samples=4 * num_need)
        mask = torch.logical_xor((ne[0] < shift), (ne[1] < shift))
        ne = ne[:, mask]
        nes = torch.cat([nes, ne[:, :num_need]], dim=-1)
    return nes


import os, sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
