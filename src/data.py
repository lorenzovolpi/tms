import hashlib
import os
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Self, Tuple

import joblib
import numpy as np
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from numba import njit
from quapy.data import LabelledCollection
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator, clone

import env
from util import split_validation


@njit
def _lookup_sample(X, S, order, left, right):
    res = []
    for j in range(S.shape[0]):
        row = S[j]
        cand = order[left[j] : right[j]]
        for i in cand:
            ok = True
            for k in range(X.shape[1]):
                if X[i, k] != row[k]:
                    ok = False
                    break
            if ok:
                res.append(i)
                break
    return res


class PreTrainedClassifier(ABC):
    GOLDEN_RATIO_MULTIPLIER = 11400714819323198485
    MULTIPLIER_64 = 0x9E3779B97F4A7C15

    def __init__(
        self,
        U_X: np.ndarray,
        U_posteriors: np.ndarray,
        V_X: np.ndarray,
        V_posteriors: np.ndarray,
        class_name: str,
        params: dict,
        default: bool,
        ms_ignore: bool,
    ):
        self.class_name: str = class_name
        self.params: dict = params
        self.default: bool = default
        self.ms_ignore: bool = ms_ignore
        self.X = np.ascontiguousarray(np.vstack([U_X, V_X]))
        self.P = np.vstack([U_posteriors, V_posteriors])
        self.row_dtype = np.dtype((np.void, self.X.dtype.itemsize * self.X.shape[1]))
        self.buckets = self._build_keys(self.X)

    @classmethod
    def _hash_params(cls, params: dict):
        params_str = cls._get_params_string(params)
        return hashlib.sha256(params_str.encode()).hexdigest()[:64]

    @classmethod
    def _get_params_string(cls, params: dict):
        params_str = "[" + ";".join([f"{k}={v}" for k, v in params.items()]) + "]"
        return params_str

    @classmethod
    def _get_name(cls, default: bool, class_name: str, params: dict):
        if default:
            return class_name
        params_str = cls._get_params_string(params)
        name = f"{class_name}_{params_str}"
        return name

    @classmethod
    def _get_full_name(cls, default: bool, class_name: str, params: dict):
        if default:
            return class_name
        full_name = f"{class_name}_{cls._hash_params(params)}"
        return full_name

    @property
    def name(self):
        return self._get_name(self.default, self.class_name, self.params)

    @property
    def full_name(self):
        return self._get_full_name(self.default, self.class_name, self.params)

    def _fp64_rows(self, X):
        row_bytes = X.dtype.itemsize * X.shape[1]
        if row_bytes % 8 != 0:
            # padding a multipli di 8 byte
            pad = 8 - (row_bytes % 8)
            Xb = np.ascontiguousarray(X.view(np.uint8))
            Xb = np.pad(Xb, ((0, 0), (0, pad)), mode="constant")
            U = Xb.view(np.uint64)
        else:
            U = X.view(np.uint64)

        U = U.reshape(X.shape[0], -1)

        # mixing semplice e veloce
        h = np.bitwise_xor.reduce(U * np.uint64(self.GOLDEN_RATIO_MULTIPLIER), axis=1)
        h ^= np.uint64(self.MULTIPLIER_64)
        return h

    def _build_keys_old(self, X: np.ndarray):
        fps = self._fp64_rows(X)
        buckets = defaultdict(list)
        for i, h in enumerate(fps):
            buckets[h].append(i)
        return buckets

    def _lookup_old(self, data, fp):
        _ids = self.buckets[fp]
        _hits = self.X[_ids, :]
        _eq_idx = np.nonzero((_hits == data).all(axis=-1))[0][0]
        return _ids[_eq_idx]

    def predict_proba_old(self, X: np.ndarray) -> np.ndarray:
        Xc = np.ascontiguousarray(X)
        fps = self._fp64_rows(Xc)
        P_idx = np.vectorize(self._lookup, signature="(m),()->()")(Xc, fps)
        return self.P[P_idx, :]

    def _build_keys(self, X: np.ndarray):
        fps = self._fp64_rows(X)
        order = np.argsort(fps)
        fps_sorted = fps[order]
        return order, fps_sorted

    def predict_proba_old2(self, X: np.ndarray) -> np.ndarray:
        sentinel = [0, X.shape[1] - 1, int(X.shape[1] / 2)]
        Xc = np.ascontiguousarray(X)
        fps = self._fp64_rows(Xc)
        uniq_fps, inv_uniq_fps = np.unique(fps, return_inverse=True)
        order, hx_sorted = self.buckets
        left = np.searchsorted(hx_sorted, uniq_fps, side="left")
        right = np.searchsorted(hx_sorted, uniq_fps, side="right")

        cands = [order[l:r] for l, r in zip(left, right)]

        # verifica esatta (anti-collisione) e ritorno liste di indici
        idx_list = []
        for j in range(Xc.shape[0]):
            cand = cands[inv_uniq_fps[j]]
            for k in sentinel:
                cand = cand[self.X[cand, k] == Xc[j, k]]
                if cand.size == 1:
                    break
            if cand.size > 1:
                # verifica vettoriale sui candidati
                mask = (self.X[cand] == Xc[j]).all(axis=1)
                cand = cand[mask]
            # print(cand, mask, cand[mask])
            idx_list.append(cand[0])
        return self.P[idx_list, :]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xc = np.ascontiguousarray(X)
        fps = self._fp64_rows(Xc)
        order, hx_sorted = self.buckets
        left = np.searchsorted(hx_sorted, fps, side="left")
        right = np.searchsorted(hx_sorted, fps, side="right")
        idx_list = _lookup_sample(self.X, Xc, order, left, right)
        return self.P[idx_list, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        posteriors = self.predict_proba(X)
        return posteriors.argmax(axis=-1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)


class ClsVariant:
    def __init__(self, class_name: str, h: BaseEstimator, params: dict, ms_ignore=False):
        self.class_name: str = class_name
        self.params: dict = params
        self.default: bool = self._is_default(h, params)
        self.h: BaseEstimator = self._get_cls(h, params)
        self.ms_ignore: bool = ms_ignore

    def _is_default(self, base, params):
        _par_names = list(params.keys())
        return params == {k: v for k, v in base.get_params().items() if k in _par_names}

    def _get_cls(self, h, params):
        _h = clone(h)
        _h.set_params(**params)
        return _h


class DatasetBundle:
    def __init__(
        self,
        dataset_name: str,
        dataset_collection: str,
        n_classes: int,
        L_prevalence: np.ndarray,
        V: LabelledCollection,
        U: LabelledCollection,
    ):
        self.dataset_name = dataset_name
        self.dataset_collection = dataset_collection
        self.n_classes = n_classes
        self.L_prevalence: np.ndarray = L_prevalence
        self.V = V
        self.test_prot = UPP(
            U,
            repeats=env.NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )
        self.V1, self.V2_prot = split_validation(self.V, random_state=qp.environ["_R_SEED"])

    # def get_posteriors(self, h: BaseEstimator):
    #     # precomumpute model posteriors for validation sets
    #     self.V_posteriors = h.predict_proba(self.V.X)
    #     self.V1_posteriors = h.predict_proba(self.V1.X)
    #     self.V2_prot_posteriors = []
    #     for sample in self.V2_prot():
    #         self.V2_prot_posteriors.append(h.predict_proba(sample.X))
    #
    #     # precomumpute model posteriors for test samples
    #     self.test_prot_posteriors, self.test_prot_y_hat, self.test_prot_true_cts = [], [], []
    #     for sample in self.test_prot():
    #         P = h.predict_proba(sample.X)
    #         self.test_prot_posteriors.append(P)
    #         y_hat = np.argmax(P, axis=-1)
    #         self.test_prot_y_hat.append(y_hat)
    #         self.test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))
    #
    #     return self
    #
    # def get_true_accs(self, accs: List[Tuple[str, Callable[[np.ndarray, np.ndarray], float]]]):
    #     # compute true accs for h on dataset
    #     self.true_accs = {} if self.true_accs is None else self.true_accs
    #     missing_accs = [(acc_name, acc_fn) for acc_name, acc_fn in accs if acc_name not in self.true_accs]
    #     for acc_name, acc_fn in missing_accs:
    #         self.true_accs[acc_name] = [
    #             true_acc_from_posteriors(acc_fn, Ui, Ui_P)
    #             for Ui, Ui_P in IT.zip_longest(self.test_prot(), self.test_prot_posteriors)
    #         ]
    #
    #     if len(missing_accs) > 0:
    #         self.updated = True
    #
    #     return self
    #
    # @classmethod
    # def mock(cls, dataset_name="mock"):
    #     return DatasetBundle(dataset_name, None, None, None, test_prot=lambda: [])
    #
    # @property
    # def empty(self):
    #     return self.V is None or self.U is None
    #
    # def dump(self):
    #     fields_to_dump: list[str] = [
    #         "V_posteriors",
    #         "V1_posteriors",
    #         "V2_prot_posteriors",
    #         "test_prot_posteriors",
    #         "test_prot_y_hat",
    #         "test_prot_true_cts",
    #         "true_accs",
    #     ]
    #     data = {f: getattr(self, f) for f in fields_to_dump}
    #     return data
    #
    # def load(self, data: dict):
    #     return replace(self, **data)


@dataclass
class ClassifierDatasetBundle:
    dataset_name: str
    dataset_collection: str
    n_classes: int
    V_posteriors: np.ndarray
    U_posteriors: np.ndarray
    h_class_name: str
    h_params: dict
    h_default: bool
    h_ms_ignore: bool

    def load_dataset(self) -> Tuple[LabelledCollection, LabelledCollection, LabelledCollection]:
        if self.dataset_collection == "uci_binary":
            fetch_UCIBinaryDataset(self.dataset_name)
        elif self.dataset_collection == "uci_multiclass":
            fetch_UCIMulticlassDataset(self.dataset_name)
        else:
            raise ValueError(f"Unknown dataset collection: {self.dataset_collection}")

    @classmethod
    def load(cls, bunlde_path: str) -> Self:
        return joblib.load(bunlde_path)

    @property
    def h_name(self):
        return PreTrainedClassifier._get_name(self.h_default, self.h_class_name, self.h_params)

    @property
    def h_full_name(self):
        return PreTrainedClassifier._get_full_name(self.h_default, self.h_class_name, self.h_params)

    def save(self):
        basedir = os.path.join("output", "tms", "pretrain")
        os.makedirs(basedir, exist_ok=True)
        save_path = os.path.join(basedir, f"{self.h_full_name}_{self.dataset_name}_{self.n_classes}.joblib")
        joblib.dump(self, save_path)

    def get_dataset_classifier(self) -> Tuple[DatasetBundle, PreTrainedClassifier]:
        L, V, U = self.load_dataset()
        dataset = DatasetBundle(
            self.dataset_name,
            self.dataset_collection,
            self.n_classes,
            L.prevalence(),
            V,
            U,
        )
        h = PreTrainedClassifier(
            U_X=U.X,
            U_posteriors=self.U_posteriors,
            V_X=V.X,
            V_posteriors=self.V_posteriors,
            class_name=self.h_class_name,
            params=self.h_params,
            default=self.h_default,
            ms_ignore=self.h_ms_ignore,
        )
        return h, dataset
