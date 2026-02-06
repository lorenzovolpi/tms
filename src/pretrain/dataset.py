import os
from glob import glob

import numpy as np
from quapy.data import LabelledCollection

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
