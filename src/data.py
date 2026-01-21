import hashlib
import itertools as IT
import os
from dataclasses import dataclass, fields, replace
from typing import Callable, List, Tuple

import joblib
import numpy as np
import quapy as qp
from cap.utils.commons import contingency_table, true_acc_from_posteriors
from quapy.data import LabelledCollection
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

import env
from util import split_validation


@dataclass
class DatasetBundle:
    dataset_name: str
    L: LabelledCollection
    V: LabelledCollection
    U: LabelledCollection
    L_prevalence: np.ndarray = None
    V1: LabelledCollection = None
    V2_prot: AbstractStochasticSeededProtocol = None
    test_prot: AbstractStochasticSeededProtocol = None
    V_posteriors: np.ndarray = None
    V1_posteriors: np.ndarray = None
    V2_prot_posteriors: np.ndarray = None
    test_prot_posteriors: np.ndarray = None
    test_prot_y_hat: np.ndarray = None
    test_prot_true_cts: np.ndarray = None
    true_accs: dict = None
    n_classes: int = -1
    loaded: bool = False
    updated: bool = False

    def create_sets(self):
        self.L_prevalence = self.L.prevalence()
        self.n_classes = self.L_prevalence.shape[0]

        # generate test protocol
        self.test_prot = UPP(
            self.U,
            repeats=env.NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # split validation set
        self.V1, self.V2_prot = split_validation(self.V, random_state=qp.environ["_R_SEED"])

        return self

    def get_posteriors(self, h: BaseEstimator):
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

    def get_true_accs(self, accs: List[Tuple[str, Callable[[np.ndarray, np.ndarray], float]]]):
        # compute true accs for h on dataset
        self.true_accs = {} if self.true_accs is None else self.true_accs
        missing_accs = [(acc_name, acc_fn) for acc_name, acc_fn in accs if acc_name not in self.true_accs]
        for acc_name, acc_fn in missing_accs:
            self.true_accs[acc_name] = [
                true_acc_from_posteriors(acc_fn, Ui, Ui_P)
                for Ui, Ui_P in IT.zip_longest(self.test_prot(), self.test_prot_posteriors)
            ]

        if len(missing_accs) > 0:
            self.updated = True

        return self

    @classmethod
    def mock(cls, dataset_name="mock"):
        return DatasetBundle(dataset_name, None, None, None, test_prot=lambda: [])

    @property
    def empty(self):
        return self.V is None or self.U is None


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

    @classmethod
    def mock(cls):
        return ClsVariant("mock", LogisticRegression(), {})
