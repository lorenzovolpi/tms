import os
from argparse import ArgumentParser
from collections import defaultdict

import requests
from datasets import load_dataset
from torchvision.datasets import STL10, OxfordIIITPet
from tqdm import tqdm

from data import DatasetInfo
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
        "flwrlabs/caltech101",
        "ljnlonoljpiljm/caltech256",
    ]

    for d in hf_datasets:
        print(d)
        dataset = load_dataset(d)
        print(dataset)

    print(OxfordIIITPet(root=os.path.join("data", "datasets"), download=True))
    print(STL10(root=os.path.join("data", "datasets"), download=True))


def imagenet_lt():
    def load_set(split):
        path = os.path.join("data", f"ImageNet_LT_{split}.txt")
        data = defaultdict(list)
        with open(path, "r") as f:
            for line in f.readlines():
                id, label = tuple(map(lambda s: s.strip(), line.strip().split(" ", maxsplit=1)))
                data[int(label)].append(id)

        return data

    train = load_set("train")
    val = load_set("val")
    test = load_set("test")

    train_size = sum([len(v) for _, v in train.items()])
    val_size = sum([len(v) for _, v in val.items()])
    test_size = sum([len(v) for _, v in test.items()])

    print(f"train: {train_size}, val: {val_size}, test: {test_size}\n")

    max_lbl = max([max(list(train.keys())), max(list(val.keys())), max(list(test.keys()))])

    trainval = {}
    for i in range(max_lbl + 1):
        trainval[i] = train.get(i, []) + val.get(i, [])

    trainval_stat = sorted([(k, len(v)) for k, v in trainval.items()], key=lambda x: x[1], reverse=True)

    with open("data/trainval_stat.txt", "w") as f:
        tot, tot_test = 0, 0
        _max, _min = None, None
        for k, v in trainval_stat[:200]:
            print(f"{k}: {v}", file=f)
            if _max is None:
                _max = v
            _min = v
            tot += v
            tot_test += len(test[k])
        print(f"\n{tot=}\n{tot_test=}\nIR={_max / _min}", file=f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--imagenet-lt", action="store_const", dest="domain", const="imagenet_lt")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.domain == "text":
        text()
    elif pargs.domain == "image":
        image()
    elif pargs.domain == "imagenet_lt":
        imagenet_lt()
    else:
        raise NotImplementedError
