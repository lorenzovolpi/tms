import itertools as IT
from collections import defaultdict
from typing import Callable, Iterable, Literal, Tuple

import numpy as np
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import f1, f1_macro, k_bin, k_macro, smooth, vanilla_acc
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import env
from data import ClassifierInfo, PretainInfo, load_info_paths
from method.ims import IMS
from method.tms import LEAP, RQBS, DoC, PrediQuant
from svmlight import SVMlight
from util import all_results_exist, sort_datasets_by_size


def kdey():
    return KDEyML(MLP())


# def get_cls_name(base_name: str, params: dict, is_default: bool):
#     if is_default:
#         return base_name
#
#     params_str = ";".join([f"{k}={v}" for k, v in params.items()])
#     return f"{base_name}_[{params_str}]"


def _get_classifier(h, params):
    _h = clone(h)
    _h.set_params(**params)
    return _h


def _is_h_default(base, params):
    _par_names = list(params.keys())
    return params == {k: v for k, v in base.get_params().items() if k in _par_names}


def _get_class_weights(n_classes):
    if n_classes == 2:
        _prevs = np.around(np.linspace(0, 1, 5, endpoint=False)[1:], decimals=2)
        _nprevs = np.around(1 - _prevs, decimals=2)
        _dicts = [{0: v, 1: nv} for v, nv in zip(_prevs.tolist(), _nprevs.tolist())]
        _dicts += [None, "balanced"]
        return _dicts
    else:
        _alpha = 2.0
        _x = _alpha / n_classes
        _y = (1.0 - _x) / (n_classes - 1)
        _prevs = np.eye(n_classes)
        _prevs = np.around(np.where(_prevs == 1, _x, _y), decimals=4)
        _dicts = [dict(zip(range(n_classes), _p)) for _p in _prevs.tolist()]
        return _dicts + [None, "balanced"]


def gen_classifier_classes(n_classes):
    LR_param_grid = {
        "C": np.logspace(-2, 2, 5),
        "class_weight": _get_class_weights(n_classes),
    }
    kNN_param_grid = {
        "n_neighbors": np.linspace(5, 13, 5, dtype="int"),
        "weights": ["uniform", "distance"],
    }
    SVM_param_grid = {
        "C": np.logspace(-2, 2, 5),
        "gamma": ["scale", "auto"],
        "class_weight": _get_class_weights(n_classes),
    }
    MLP_param_grid = {
        "alpha": np.around(np.logspace(-5, -1, 5), decimals=5),
        "learning_rate": ["constant", "adaptive"],
    }

    yield "LR", LogisticRegression(), LR_param_grid
    yield "kNN", KNN(), kNN_param_grid
    yield "SVM", SVC(kernel="rbf", probability=True), SVM_param_grid
    yield "MLP", MLP(), MLP_param_grid


def gen_classifiers(n_classes) -> Iterable[Tuple[BaseEstimator, ClassifierInfo]]:
    for name, base, param_grid in gen_classifier_classes(n_classes):
        _par_names = list(param_grid.keys())
        _par_combos = IT.product(*list(param_grid.values()))
        for _combo in _par_combos:
            _params = dict(zip(_par_names, _combo))
            h = _get_classifier(base, _params)
            is_default = _is_h_default(base, _params)
            yield h, ClassifierInfo(class_name=name, params=_params, default=is_default)

    # SVM-transductive classifier
    svmt = SVMlight(kernel="rbf")
    yield svmt, ClassifierInfo(class_name="SVM-t", params={}, default=True)


def gen_datasets(
    only_names=False,
) -> Iterable[tuple[str, tuple[LabelledCollection, LabelledCollection, LabelledCollection] | None]]:
    _uci_bin_native = [
        "breast-cancer",
        "german",
        "haberman",
        "ionosphere",
        "mammographic",
        "semeion",
        "sonar",
        "spambase",
        "spectf",
        "tictactoe",
        "transfusion",
        "wdbc",
        # "yeast",
    ]
    _uci_bin_names = [d for d in UCI_BINARY_DATASETS if d in _uci_bin_native]
    coll = "uci_binary"
    _sorted_bin_names = sort_datasets_by_size(coll, _uci_bin_names, fetch_UCIBinaryDataset)
    for dn in _sorted_bin_names[:5]:
        dval = None if only_names else fetch_UCIBinaryDataset(dn)
        yield dn, coll, dval
    _uci_mul_names = [d for d in UCI_MULTICLASS_DATASETS]
    coll = "uci_multiclass"
    _sorted_mul_names = sort_datasets_by_size(coll, _uci_mul_names, fetch_UCIMulticlassDataset)
    for dn in _sorted_mul_names:
        dval = None if only_names else fetch_UCIMulticlassDataset(dn)
        yield dn, coll, dval


def get_acc_names():
    return ["vanilla_accuracy", "macro-F1", "macro-K"]


def get_selection_acc(name: str, multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": (smooth(f1_macro) if multiclass else smooth(f1)),
        "macro-K": (k_macro if multiclass else k_bin),
    }[name]


def get_evaluation_acc(name: str, is_multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": (f1_macro if is_multiclass else f1),
        "macro-K": (k_macro if is_multiclass else k_bin),
    }[name]


def acc_from_ct(acc_name: str, ct: np.ndarray, type: Literal["selection", "evaluation"] = "selection") -> float:
    n_classes = ct.shape[0]
    if type == "selection":
        return get_selection_acc(acc_name, n_classes > 2)(ct)
    elif type == "evaluation":
        return get_evaluation_acc(acc_name, n_classes > 2)(ct)


def gen_acc_measure(is_multiclass: bool):
    for acc in get_acc_names():
        yield acc, get_selection_acc(acc, is_multiclass)


def gen_methods():
    _, acc = next(gen_acc_measure(True))
    yield "IMS", IMS(acc)
    yield "TMS_LEAP", LEAP(acc)
    yield "TMS_RQBS", RQBS(acc)
    # yield "TMS_RQBScap", RQBScap()
    # yield "TMS_PrediQuant", PrediQuant(acc)
    # yield "TMS_DoC", DoC(acc)


def get_classifier_names():
    mock_n_classes = 2
    return [clsf.name for clsf in gen_classifiers(mock_n_classes)]


def get_classifier_class_names():
    mock_n_classes = 2
    return [name for name, _, _ in gen_classifier_classes(mock_n_classes)]


def get_dataset_names():
    return [name for name, _, _ in gen_datasets(only_names=True)]


def get_existing_dataset_names(experiment: str, domain: str, sort=True):
    info_paths = load_info_paths(domain=domain)
    dataset_h_map = defaultdict(lambda: True)
    for path in info_paths:
        p = PretainInfo.load(path, fast=True)
        d_info, h_info = p.d_info, p.h_info
        # if not dataset_h_map[D.name]:
        #     continue
        # problem = "multiclass" if d_info.n_classes > 2 else "binary"
        _key = (d_info.name, d_info.collection)
        dataset_h_map[_key] = dataset_h_map[_key] and all_results_exist(
            p.domain, d_info.name, h_info.full_name, get_method_names(), get_acc_names(), experiment
        )

    datasets = [dc for dc, all_exist in dataset_h_map.items() if all_exist]
    dataset_names, dataset_colls = tuple(map(lambda x: list(x), zip(*datasets)))
    if sort:
        return sort_datasets_by_size(dataset_colls, dataset_names)
    else:
        return dataset_names


def get_method_names():
    names = [m for m, _ in gen_methods()]

    return names
