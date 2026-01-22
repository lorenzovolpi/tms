import itertools as IT
from typing import Iterable

import numpy as np
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import f1, f1_macro, k_bin, k_macro, smooth, vanilla_acc
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import env
from data import ClsVariant, DatasetBundle
from method.ims import IMS
from method.tms import LEAP, RQBS, DoC, PrediQuant, RQBScap
from svmlight import SVMlight
from util import sort_datasets_by_size


def kdey():
    return KDEyML(MLP())


# def get_cls_name(base_name: str, params: dict, is_default: bool):
#     if is_default:
#         return base_name
#
#     params_str = ";".join([f"{k}={v}" for k, v in params.items()])
#     return f"{base_name}_[{params_str}]"


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


def gen_classifiers(n_classes) -> Iterable[ClsVariant]:
    for name, base, param_grid in gen_classifier_classes(n_classes):
        _par_names = list(param_grid.keys())
        _par_combos = IT.product(*list(param_grid.values()))
        for _combo in _par_combos:
            _params = dict(zip(_par_names, _combo))
            yield ClsVariant(class_name=name, h=base, params=_params)

    # SVM-transductive classifier
    yield ClsVariant(
        class_name="SVM-t",
        h=SVMlight(kernel="rbf"),
        params={},
        ms_ignore=False,
        dumpable_h=False,
    )


def gen_datasets(
    only_names=False,
) -> Iterable[tuple[str, tuple[LabelledCollection, LabelledCollection, LabelledCollection] | None]]:
    if env.PROBLEM == "binary":
        # _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        # _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
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
        _uci_names = [d for d in UCI_BINARY_DATASETS if d in _uci_bin_native]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names[:5]:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif env.PROBLEM == "multiclass":
        # _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_skip = []
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        for dataset_name in _sorted_uci_names:
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


def gen_acc_measure():
    multiclass = env.PROBLEM == "multiclass"
    yield "vanilla_accuracy", vanilla_acc
    yield "macro-F1", (smooth(f1_macro) if multiclass else smooth(f1))
    # yield "macro-K", (k_macro if multiclass else k_bin)


def gen_methods(clsf: ClsVariant, D: DatasetBundle):
    yield "IMS", IMS(clsf, D), D.V, D.V_posteriors
    yield "TMS_LEAP", LEAP(clsf, D), D.V, D.V_posteriors
    yield "TMS_RQBS", RQBS(clsf, D), D.V, D.V_posteriors
    # yield "TMS_RQBScap", RQBScap(clsf, D), D.V, D.V_posteriors
    # yield "TMS_PrediQuant", PrediQuant(clsf, D), D.V1, D.V1_posteriors
    # yield "TMS_DoC", DoC(clsf, D), D.V1, D.V1_posteriors


def get_classifier_names():
    mock_n_classes = 2
    return [clsf.name for clsf in gen_classifiers(mock_n_classes)]


def get_classifier_class_names():
    mock_n_classes = 2
    return [name for name, _, _ in gen_classifier_classes(mock_n_classes)]


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def get_all_dataset_names():
    _orig_prob = env.PROBLEM
    all_datasets = []
    for _prob in env._valid_problems:
        env.PROBLEM = _prob
        all_datasets.extend(get_dataset_names())
    env.PROBLEM = _orig_prob

    return sort_datasets_by_size(all_datasets)


def get_acc_names():
    return [acc_name for acc_name, _ in gen_acc_measure()]


def get_method_names():
    mock_clsf = ClsVariant.mock()
    mock_D = DatasetBundle.mock()

    names = [m for m, _, _, _ in gen_methods(mock_clsf, mock_D)]

    return names
