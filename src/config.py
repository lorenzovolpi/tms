import hashlib
import itertools as IT
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import f1, f1_macro, smooth, vanilla_acc
from cap.models.cont_table import LEAP, O_LEAP, NaiveCAP
from cap.models.utils import OracleQuantifier
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import env
from svmlight import SVMlight
from util import sort_datasets_by_size, split_validation

_toggle = {
    "vanilla": True,
    "f1": False,
}


@dataclass
class DatasetBundle:
    L_prevalence: np.ndarray
    V: LabelledCollection
    U: LabelledCollection
    V1: LabelledCollection = None
    V2_prot: AbstractStochasticSeededProtocol = None
    test_prot: AbstractStochasticSeededProtocol = None
    V_posteriors: np.ndarray = None
    V1_posteriors: np.ndarray = None
    V2_prot_posteriors: np.ndarray = None
    test_prot_posteriors: np.ndarray = None
    test_prot_y_hat: np.ndarray = None
    test_prot_true_cts: np.ndarray = None

    def create_bundle(self, h: BaseEstimator):
        # generate test protocol
        self.test_prot = UPP(
            self.U,
            repeats=env.NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # split validation set
        self.V1, self.V2_prot = split_validation(self.V)

        # precomumpute model posteriors for validation sets
        self.V_posteriors = h.predict_proba(self.V.X)
        self.V1_posteriors = h.predict_proba(self.V1.X)
        self.V2_prot_posteriors = []
        for sample in self.V2_prot():
            self.V2_prot_posteriors.append(h.predict_proba(sample.X))

        # precomumpute model posteriors for test samples
        self.test_prot_posteriors, self.test_prot_y_hat, self.test_prot_true_cts = [], [], []
        for sample in self.test_prot():
            P = h.predict_proba(sample.X)
            self.test_prot_posteriors.append(P)
            y_hat = np.argmax(P, axis=-1)
            self.test_prot_y_hat.append(y_hat)
            self.test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))

        return self

    @classmethod
    def mock(cls):
        return DatasetBundle(None, None, None, test_prot=lambda: [])


class ClsVariant:
    def __init__(self, class_name: str, h: BaseEstimator, params: dict, ms_ignore=False):
        self.class_name: str = class_name
        self.base = h
        self.params: dict = params
        self.default: bool = self._is_default(h, params)
        self.h: BaseEstimator = self._get_cls(h, params)
        self.ms_ignore: bool = ms_ignore

        self._set_names()

    def _is_default(self, base, params):
        _par_names = list(params.keys())
        return params == {k: v for k, v in base.get_params().items() if k in _par_names}

    def _get_cls(self, h, params):
        _h = clone(h)
        _h.set_params(**params)
        return _h

    def _set_names(self):
        if self.default:
            self.file_name = self.name = self.class_name
            return

        def hash_params(params_str):
            return hashlib.sha256(params_str.encode()).hexdigest()[:64]

        params_str = "[" + ";".join([f"{k}={v}" for k, v in self.params.items()]) + "]"
        self.name: str = f"{self.class_name}_{params_str}"
        self.file_name: str = f"{self.class_name}_{hash_params(params_str)}"

    def clone(self):
        return ClsVariant(self.class_name, self.base, self.params, self.ms_ignore)


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
    yield ("macro-F1", smooth(f1_macro)) if multiclass else ("F1", smooth(f1))


def gen_CAP_cont_table(h, acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    # yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True)
    yield "O-LEAP(KDEy)", O_LEAP(acc_fn, kdey())


def gen_methods_with_oracle(h, acc_fn, D: DatasetBundle):
    oracle_q = OracleQuantifier([ui for ui in D.test_prot()])
    yield "LEAP(oracle)", LEAP(acc_fn, oracle_q, reuse_h=h, log_true_solve=True)
    # yield "O-LEAP(oracle)", OCE(acc_fn, oracle_q, reuse_h=h, optim_method="SLSQP")


def gen_CAP_methods(h, D, with_oracle=False):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, D.V, D.V_posteriors
    if with_oracle:
        for name, method in gen_methods_with_oracle(h, acc_fn, D):
            yield name, method, D.V, D.V_posteriors


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


def get_CAP_method_names(with_oracle=False):
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()

    names = [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]

    if with_oracle:
        names += [m for m, _ in gen_methods_with_oracle(mock_h, mock_acc_fn, mock_D)]

    return names
