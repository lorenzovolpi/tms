import os
import subprocess as sb
from collections import defaultdict
from glob import glob

import numpy as np
import quapy as qp
import requests
from quapy.data import LabelledCollection
from sklearn.model_selection import train_test_split
from tqdm import tqdm

BASEDIR = os.path.join("output", "tms", "datasets")


def get_dataset_path(domain: str, dataset_name: str, model_name: str | None):
    if model_name is None or model_name == "*":
        return glob(os.path.join(BASEDIR, f"{domain}_{dataset_name}_*.npz"))[0]
    return os.path.join(BASEDIR, f"{domain}_{dataset_name}_{model_name}.npz")


def save_dataset(domain, dataset_name, model_name, classes, train_prev, embeds, labels):
    path = get_dataset_path(domain, dataset_name, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = dict(
        classes=classes,
        L_prevalence=train_prev,
        validation_X=embeds["validation"],
        validation_y=labels["validation"],
        test_X=embeds["test"],
        test_y=labels["test"],
    )
    np.savez_compressed(path, **d)


def load_dataset(domain, dataset_name, model_name=None):
    path = get_dataset_path(domain, dataset_name, model_name)
    _data = np.load(path)
    _classes = _data["classes"]
    L_prevalence = _data["L_prevalence"]
    V = LabelledCollection(instances=_data["validation_X"], labels=_data["validation_y"], classes=_classes)
    U = LabelledCollection(instances=_data["test_X"], labels=_data["test_y"], classes=_classes)
    return L_prevalence, V, U


def download_imagenet_split(name):
    urls = {}
    with open(os.path.join("data", "imagenet_urls.txt"), "r") as f:
        for line in f.readlines():
            _split, _url = tuple(map(lambda s: s.strip(), line.strip().split(" ", maxsplit=1)))
            urls[_split] = _url
    url = urls[name]

    dest_dir = os.path.join("data", "datasets", "imagenet-1k")
    os.makedirs(dest_dir, exist_ok=True)

    filename = url.split("/")[-1]
    path = os.path.join(dest_dir, filename)

    if os.path.exists(path):
        return

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Controlla se ci sono errori HTTP

            with open(path, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(chunk_size=8192),
                    desc=f"{name}",
                    total=int(r.headers.get("content-length", 0)) // 8192,
                ):
                    if chunk:  # filtra i keep-alive chunks
                        f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Errore durante il download di {name}: {e}")


def extract_imagenet_split(name):
    filename = {
        "train": "ILSVRC2012_img_train.tar",
        "val": "ILSVRC2012_img_val.tar",
    }[name]

    dest_dir = os.path.join("data", "datasets", "imagenet-1k")
    os.makedirs(dest_dir, exist_ok=True)

    split_dir = os.path.join(dest_dir, name)
    if os.path.exists(split_dir) and len(glob(os.path.join(split_dir, "*"))) > 1:
        return
    os.makedirs(split_dir, exist_ok=True)

    tar_path = os.path.join(dest_dir, filename)
    if not os.path.exists(tar_path):
        download_imagenet_split(name)

    sb.run(f"tar -xf {tar_path} -C {split_dir}", shell=True)
    sb.run(f"rm {tar_path}", shell=True)
    for path in tqdm(glob(os.path.join(split_dir, "*.tar")), desc=f"extracting {name}"):
        class_dir = os.path.join(split_dir, os.path.basename(path).split(".")[0])
        os.makedirs(class_dir, exist_ok=True)
        sb.run(f"tar -xf {path} -C {class_dir}", shell=True)
        sb.run(f"rm {path}", shell=True)


def extract_imagenet():

    extract_imagenet_split("train")
    extract_imagenet_split("val")


def build_imagenet_lt():
    def load_set(split) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join("data", f"ImageNet_LT_{split}.txt")
        X, y = [], []
        with open(path, "r") as f:
            for line in f.readlines():
                id, label = tuple(map(lambda s: s.strip(), line.strip().split(" ", maxsplit=1)))
                X.append(id)
                y.append(int(label))

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y

    train_X, train_y = load_set("train")
    val_X, val_y = load_set("val")
    test_X, test_y = load_set("test")

    trainval_X = np.concatenate([train_X, val_X])
    trainval_y = np.concatenate([train_y, val_y])

    ns_train_X, ns_val_X, ns_train_y, ns_val_y = train_test_split(
        trainval_X,
        trainval_y,
        test_size=20000,
        random_state=qp.environ["_R_SEED"],
        stratify=trainval_y,
    )

    for x in test_X[:20]:
        print(x)


if __name__ == "__main__":
    qp.environ["_R_SEED"] = 0
    download_imagenet_split("val")
