import os
import subprocess as sb
from collections import defaultdict
from glob import glob

import numpy as np
import quapy as qp
import requests
from datasets import Dataset, DatasetDict, Image, load_from_disk
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


def extract_imagenet_train():
    name = "train"
    filename = "ILSVRC2012_img_train.tar"
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


def extract_imagenet_val():
    name = "val"
    filename = "ILSVRC2012_img_val.tar"
    dest_dir = os.path.join("data", "datasets", "imagenet-1k")
    os.makedirs(dest_dir, exist_ok=True)

    split_dir = os.path.join(dest_dir, name)
    if os.path.exists(split_dir) and len(glob(os.path.join(split_dir, "*"))) > 1:
        return
    os.makedirs(split_dir, exist_ok=True)

    lt_tes_path = os.path.join("data", "ImageNet_LT_test.txt")
    class_map = {}
    with open(lt_tes_path, "r") as f:
        for line in f.readlines():
            dest_path, _ = tuple(map(lambda s: s.strip(), line.strip().split(" ", maxsplit=1)))
            parts = dest_path.split("/")
            class_map[parts[2]] = parts[1]

    tar_path = os.path.join(dest_dir, filename)
    if not os.path.exists(tar_path):
        download_imagenet_split(name)

    sb.run(f"tar -xf {tar_path} -C {split_dir}", shell=True)
    sb.run(f"rm {tar_path}", shell=True)
    for path in tqdm(glob(os.path.join(split_dir, "*.JPEG")), desc=f"fixing {name}"):
        img_name = os.path.basename(path)
        class_name = class_map[img_name]
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        dest_path = os.path.join(class_dir, img_name)
        os.rename(path, dest_path)


def get_imagenet():
    extract_imagenet_train()
    extract_imagenet_val()


def build_imagenet_lt(val_size=20000):
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

    source_dir = os.path.join("data", "datasets", "imagenet-1k")
    dest_dir = os.path.join("data", "datasets", "imagenet-lt")
    # if os.path.exists(dest_dir):
    #     return
    os.makedirs(dest_dir, exist_ok=True)

    get_imagenet()

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

    def fix_image_path(p):
        return os.path.join(source_dir, p)

    train_dict = {"image": list(map(fix_image_path, ns_train_X.tolist())), "label": ns_train_y.tolist()}
    val_dict = {"image": list(map(fix_image_path, ns_val_X.tolist())), "label": ns_val_y.tolist()}
    test_dict = {"image": list(map(fix_image_path, test_X.tolist())), "label": test_y.tolist()}

    train_ds = Dataset.from_dict(train_dict).cast_column("image", Image())
    val_ds = Dataset.from_dict(val_dict).cast_column("image", Image())
    test_ds = Dataset.from_dict(test_dict).cast_column("image", Image())

    dataset = DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )

    dataset.save_to_disk(dest_dir)


if __name__ == "__main__":
    qp.environ["_R_SEED"] = 0
    build_imagenet_lt()

    ds = load_from_disk(os.path.join("data", "datasets", "imagenet-lt"))
    print(ds)
    print(type(ds["train"]["image"][0]))
