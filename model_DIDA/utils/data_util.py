import os
import torch
from .mutils import seed_everything
from util_hee import remove_duplicated_edges


def mkdirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def prepare_dir(output_folder):
    mkdirs(output_folder)
    log_folder = mkdirs(output_folder)
    return log_folder


def select_by_field(edges, fields=[0, 1]):
    # field [0,1,2,3,4]
    res = []
    for f in fields:
        e = edges[edges[:, 4] == f]
        res.append(e)
    edges = torch.cat(res, dim=0)
    res = []
    for i in range(16):
        e = edges[edges[:, 2] == i]
        e = e[:, :2]
        res.append(e)
    edges = res
    return edges


def select_by_venue(edges, venues=[0, 1]):
    # venue [0-21]
    res = []
    for f in venues:
        e = edges[edges[:, 3] == f]
        res.append(e)
    edges = torch.cat(res, dim=0)
    res = []
    for i in range(16):
        e = edges[edges[:, 2] == i]
        e = e[:, :2]
        res.append(e)
    edges = res
    return edges


def load_data(args, dataset=None):
    seed_everything(0)
    if dataset is None:
        dataset = args.dataset

    if dataset == "collab":
        from ..data_configs.collab import (
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
        from ..data_configs.yelp import (
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
        from ..data_configs.synthetic import (
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
        from preprocess_dict_from_dgl import PreprocessDictFromDGL

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
        from preprocess_dict_from_dgl import PreprocessDictFromDGL

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
        from preprocess_dict_from_dgl import PreprocessDictFromDGL

        data = PreprocessDictFromDGL("raw_data/WikiElec", "data/WikiElec").graph_dict
        args.dataset = dataset
        args.length = len(data["train"]["pedges"])
        args.vallength = 1
        args.testlength = int(args.length * 0.2)
        args.nfeat = data["x"].shape[1]
        args.num_nodes = len(data["x"])
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")
    print(f"Loading dataset {dataset}")
    print(
        f"Adding uniform edges features having the same shape of x in data to dataset {dataset}"
    )

    data["train"]["weights"] = []
    for t, edge_tensor in enumerate(data["train"]["pedges"]):
        edge_tensor = remove_duplicated_edges(edge_tensor)
        weights_t = {}
        for i, j in zip(edge_tensor[0], edge_tensor[1]):
            weights_t[(int(i), int(j))] = 1.0
            weights_t[(int(j), int(i))] = 1.0

        data["train"]["weights"].append(weights_t)
        data["train"]["pedges"][t] = edge_tensor

    return args, data
