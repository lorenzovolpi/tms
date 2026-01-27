import os

import numpy as np
from quapy.data import LabelledCollection

BASEDIR = os.path.join("output", "tms", "datasets")


def get_dataset_path(domain, dataset_name, model_name):
    return os.path.join(BASEDIR, f"{domain}_{dataset_name}_{model_name}.npz")


def save_sentiment(dataset_name, model_name, classes, train_prev, val_hidden_states, val_y, test_hidden_states, test_y):
    path = get_dataset_path("sentiment", dataset_name, model_name)
    d = dict(
        classes=classes,
        L_prevalence=train_prev,
        validation_X=val_hidden_states,
        validation_y=val_y,
        test_X=test_hidden_states,
        test_y=test_y,
    )
    np.savez_compressed(path, **d)


def load_sentiment(dataset_name, model_name):
    path = get_dataset_path("sentiment", dataset_name, model_name)
    _data = np.load(path)
    _classes = _data["classes"]
    L_prevalence = _data["L_prevalence"]
    V = LabelledCollection(instances=_data["validation_X"], labels=_data["validation_y"], classes=_classes)
    U = LabelledCollection(instances=_data["test_X"], labels=_data["test_y"], classes=_classes)
    return L_prevalence, V, U
