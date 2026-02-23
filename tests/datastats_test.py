import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from data import DatasetInfo
from pretrain.dataset import get_local_hf_dataset
from pretrain.text import get_dataset


def text():
    datasets = [
        ("stanfordnlp/imdb", 2),
        ("stanfordnlp/sst2", 2),
        ("fancyzhx/yelp_polarity", 2),
        ("fancyzhx/ag_news", 4),
        ("fancyzhx/dbpedia_14", 14),
        ("community-datasets/yahoo_answers_topics", 10),
    ]

    for d, n_classes in datasets:
        d_info = DatasetInfo(d, "text", n_classes)
        dataset = get_dataset(d_info)
        tot = 0
        for name, split in dataset.items():
            print(f"{d}@{name}: {split.shape[0]}")
            tot += split.shape[0]
        print(f"{d}: {tot}\n")


def image():
    hf_datasets = [
        # "mnist",
        # "cifar10",
        # "cifar100",
    ]
    local_datasets = [
        # "caltech256",
        "imagenet-lt",
        "imagenet-lt200",
        "imagenet-lt100",
        # "cifar10-lt",
        # "cifar100-lt",
        # "food101-lt",
        # "dbpedia_14-lt",
        # "ag_news-lt",
        # "yelp_reviews",
        # "rcv1-v2"
    ]

    for d in hf_datasets:
        print(d)
        dataset = load_dataset(d)
        labels = np.unique(dataset["train"]["label"])
        print(labels)

    for d in local_datasets:
        print(d)
        dataset = get_local_hf_dataset(d)
        labels = np.unique(dataset["train"]["label"])
        print(dataset)
        print(labels)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.domain == "text":
        text()
    elif pargs.domain == "image":
        image()
    else:
        raise NotImplementedError
