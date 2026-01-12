import functools
import itertools as IT
import logging
import os
from time import time

import cap
import numpy as np
import pandas as pd
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.models.base import ClassifierAccuracyPrediction
from cap.models.cont_table import CAPContingencyTable
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.protocol import UPP

import env


def fit_or_switch(method: ClassifierAccuracyPrediction, V, V_posteriors, acc_fn, is_fit):
    if hasattr(method, "switch"):
        method, t_train = method.switch(acc_fn), None
        if not is_fit:
            tinit = time()
            method.fit(V, V_posteriors)
            t_train = time() - tinit
        return method, t_train
    elif hasattr(method, "switch_and_fit"):
        tinit = time()
        method = method.switch_and_fit(acc_fn, V, V_posteriors)
        t_train = time() - tinit
        return method, t_train
    else:
        ValueError("invalid method")


def get_ct_predictions(method: ClassifierAccuracyPrediction, test_prot, test_prot_posteriors):
    tinit = time()
    if isinstance(method, CAPContingencyTable):
        estim_accs, estim_cts = method.batch_predict(test_prot, test_prot_posteriors, get_estim_cts=True)
    else:
        estim_accs, estim_cts = (
            method.batch_predict(test_prot, test_prot_posteriors),
            None,
        )
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, estim_cts, t_test_ave


def get_logger(id="quacc"):
    _name = f"{id}_log"
    _path = os.path.join(cap.env["OUT_DIR"], f"{id}.log")
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        fh = logging.FileHandler(_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%b %d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return np.around(prev[1:], decimals=4).tolist()
    else:
        return float(np.around(prev, decimals=4)[-1])


def timestamp(t_train: float, t_test_ave: float) -> str:
    t_train = round(t_train, ndigits=3)
    t_test_ave = round(t_test_ave, ndigits=3)
    return f"{t_train=}s; {t_test_ave=}s"


def get_test_prot(U: LabelledCollection, repeats=1000, sample_size=None, return_type="labelled_collection"):
    return UPP(
        U,
        repeats=repeats,
        sample_size=sample_size,
        random_state=qp.environ["_R_SEED"],
        return_type=return_type,
    )


def split_validation(V: LabelledCollection, ratio=0.6, repeats=100, sample_size=None):
    v_train, v_val = V.split_stratified(ratio, random_state=qp.environ["_R_SEED"])
    val_prot = UPP(v_val, repeats=repeats, sample_size=sample_size, return_type="labelled_collection")
    return v_train, val_prot


def is_excluded(classifier, dataset, method, acc):
    return False


def local_path(dataset_name, cls_name, method_name, acc_name, experiment=None):
    base_dir = env.root_dir if experiment is None else os.path.join(env.root_dir, experiment)
    parent_dir = os.path.join(base_dir, env.PROBLEM, acc_name, dataset_name, method_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{cls_name}.json")


def all_exist_pre_check(dataset_name, cls_name, method_names, acc_names, experiment=None):
    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        if is_excluded(cls_name, dataset_name, method, acc):
            continue
        path = local_path(dataset_name, cls_name, method, acc, experiment=experiment)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def decorate_dataset(dataset: str):
    return r"\textsf{" + dataset + r"}"


def one_hot(y: np.ndarray, n_classes: int | None = None):
    assert isinstance(y, np.ndarray) and y.ndim == 1, (
        f"Function one_hot expects a numpy.ndarray of dimension 1; found and array of dimension {y.ndim}"
    )

    if n_classes is None:
        n_classes = max(np.unique(y).shape[0], np.max(y) + 1)

    _eye = np.eye(n_classes)
    return _eye[y, :]


def sort_datasets_by_size(dataset_names: list[str], descending=True):
    @functools.lru_cache(maxsize=len(UCI_BINARY_DATASETS) + len(UCI_MULTICLASS_DATASETS))
    def get_dataset_size(name):
        if name in UCI_BINARY_DATASETS:
            L, V, U = fetch_UCIBinaryDataset(name)
        elif name in UCI_MULTICLASS_DATASETS:
            L, V, U = fetch_UCIMulticlassDataset(name)
        return len(L) + len(V) + len(U)

    datasets = [(d, get_dataset_size(d)) for d in dataset_names]
    datasets.sort(key=(lambda d: d[1]), reverse=descending)
    return [d for (d, _) in datasets]


def gen_method_df(df_len, **data):
    data = data | {k: [v] * df_len for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data, orient="columns")
