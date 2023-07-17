import os
import random
import torch
from loguru import logger
from utils_main import remove_duplicated_edges, seed_everything


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def load_data(args, dataset=None):
    seed_everything(0)
    if dataset is None:
        dataset = args.dataset

    if dataset == "collab":
        from dataset_loader.collab import (
            testlength,
            vallength,
            length,
            split,
            processed_datafile,
        )

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.split = split
        data = torch.load(f"{processed_datafile}-{split}")
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    elif dataset == "yelp":
        from dataset_loader.yelp import (
            testlength,
            vallength,
            length,
            split,
            processed_datafile,
            shift,
            num_nodes,
        )

        args.dataset = dataset
        args.testlength = testlength
        args.vallength = vallength
        args.length = length
        args.split = split
        args.shift = shift
        args.num_nodes = num_nodes
        data = torch.load(f"{processed_datafile}-{split}")
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    elif "synthetic" in dataset:
        from dataset_loader.synthetic import (
            testlength,
            vallength,
            synthetic_file,
            P,
            SIGMA,
            TEST_P,
            TEST_SIGMA,
        )

        args.testlength = testlength
        args.vallength = vallength
        P = dataset.split("-")
        P = float(P[-1]) if len(P) > 1 else 0.6
        args.dataset = f"synthetic-{P}"
        args.P = P
        args.SIGMA = SIGMA
        args.TEST_P = TEST_P
        args.TEST_SIGMA = TEST_SIGMA
        datafile = f"{synthetic_file}-{P,SIGMA,TEST_P,TEST_SIGMA}"
        data = torch.load(datafile)
        args.nfeat = data["x"][0].shape[1]
        args.num_nodes = len(data["x"][0])
        args.length = len(data["x"])

    elif "bitcoin" == dataset:
        from dataset_loader.preprocess_dict_from_dgl import PreprocessDictFromDGL

        data = PreprocessDictFromDGL(
            "raw_data/BitcoinAlpha", "data/BitcoinAlpha"
        ).graph_dict
        args.dataset = dataset
        args.length = len(data["train"]["pedges"])
        args.vallength = 1
        args.testlength = int(args.length * 0.2)
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    elif "redditbody" == dataset:
        from dataset_loader.preprocess_dict_from_dgl import PreprocessDictFromDGL

        data = PreprocessDictFromDGL(
            "raw_data/RedditBody", "data/RedditBody"
        ).graph_dict
        args.dataset = dataset
        args.length = len(data["train"]["pedges"])
        args.vallength = 1
        args.testlength = int(args.length * 0.2)
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])

    elif "wikielec" == dataset:
        from dataset_loader.preprocess_dict_from_dgl import PreprocessDictFromDGL

        data = PreprocessDictFromDGL("raw_data/WikiElec", "data/WikiElec").graph_dict
        args.dataset = dataset
        args.length = len(data["train"]["pedges"])
        args.vallength = 1
        args.testlength = int(args.length * 0.2)
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")
    logger.info(f"Loading dataset {dataset}")
    logger.info(
        f"Adding uniform edges features having the same shape of x in data to dataset {dataset}"
    )

    data["train"]["weights"] = []
    for t, edge_tensor in enumerate(data["train"]["pedges"]):
        edge_tensor = remove_duplicated_edges(edge_tensor)
        weights_t = {}
        for i, j in zip(edge_tensor[0], edge_tensor[1]):
            weights_t[(int(i), int(j))] = 1.0
            # weights_t[(int(j), int(i))] = 1.0

        data["train"]["weights"].append(weights_t)
        data["train"]["pedges"][t] = edge_tensor

    return args, data


def aggregate_by_time(raw_edges, time_aggregation):
    """
    Parameters
    ----------
    raw_edges
        list of the edges
    time_aggregation
        time step size in seconds

    Returns
    -------
    list of edges in the single time step
    """
    times = [int(re["time"] // time_aggregation) for re in raw_edges]

    min_time, max_time = min(times), max(times)
    times = [t - min_time for t in times]
    time_steps = max_time - min_time + 1
    seperated_edges = [[] for _ in range(time_steps)]

    for i, edge in enumerate(raw_edges):
        t = times[i]
        seperated_edges[t].append(edge)

    return seperated_edges


# this function guarantees unique edges
# use latest edge in single time step
def generate_undirected_edges(directed_edges):
    """
    Parameters
    ----------
    directed_edges
        directional edges

    Returns
    -------
    undirectional edges
    """

    edges_dict = {}
    for edge in directed_edges:
        e = (edge["from"], edge["to"])
        wt = edges_dict.get(e)

        if wt:
            _, t = wt
            if edge["time"] > t:
                edges_dict[e] = edge["weight"], edge["time"]
        else:
            edges_dict[e] = edge["weight"], edge["time"]

    undirected_edges = []
    for edge in edges_dict:
        if (edge[1], edge[0]) in edges_dict:
            if edge[0] > edge[1]:
                continue

            weight1, time1 = edges_dict[edge]
            weight2, time2 = edges_dict[(edge[1], edge[0])]

            weight = weight1 if time1 >= time2 else weight2

            undirected_edges.append(
                {
                    "from": edge[0],
                    "to": edge[1],
                    "weight": weight,
                    "original": time1 >= time2,
                }
            )
            undirected_edges.append(
                {
                    "from": edge[1],
                    "to": edge[0],
                    "weight": weight,
                    "original": time1 < time2,
                }
            )

        else:
            weight, _ = edges_dict[edge]
            undirected_edges.append(
                {"from": edge[0], "to": edge[1], "weight": weight, "original": True}
            )
            undirected_edges.append(
                {"from": edge[1], "to": edge[0], "weight": weight, "original": False}
            )

    return undirected_edges


def negative_sampling(adj_list):
    """
    Parameters
    ----------
    adj_list
        directed adjacency list on single time step

    Returns
    -------
    sampled undirected non_edge list from given adjacency list
    """
    num_nodes = len(adj_list)
    non_edges = []

    for node, neighbors in enumerate(adj_list):
        neg_dests = []
        while len(neg_dests) < len(neighbors):
            dest = random.randint(0, num_nodes - 1)
            if dest in [node] + neighbors + neg_dests:
                continue
            neg_dests.append(dest)
            non_edges.append([node, dest])

    return non_edges + [non_edge[::-1] for non_edge in non_edges]
