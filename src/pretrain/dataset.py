import os
from glob import glob

import numpy as np
from quapy.data import LabelledCollection

BASEDIR = os.path.join("output", "tms", "datasets")


def get_dataset_path(domain: str, dataset_name: str, model_name: str | None):
    if model_name is None or model_name == "*":
        return glob(os.path.join(BASEDIR, f"{domain}_{dataset_name}_*.npz"))[0]
    return os.path.join(BASEDIR, f"{domain}_{dataset_name}_{model_name}.npz")


def save_dataset(domain, dataset_name, model_name, classes, train_prev, embeds, logits, labels):
    path = get_dataset_path(domain, dataset_name, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    val_hidden_states = embeds["validation"]
    val_logits = logits["validation"]
    val_y = labels["validation"]
    test_hidden_states = embeds["test"]
    test_logits = logits["test"]
    test_y = labels["test"]
    d = dict(
        classes=classes,
        L_prevalence=train_prev,
        validation_X=val_hidden_states,
        validation_logits=val_logits,
        validation_y=val_y,
        test_X=test_hidden_states,
        test_logits=test_logits,
        test_y=test_y,
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
