import hashlib
import json
import os
import pickle
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
from glob import glob
from typing import Callable, Literal, Self, Tuple

import joblib
import numpy as np
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.utils.commons import contingency_table
from numba import njit
from quapy.data import LabelledCollection
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator, clone

import env
from util import split_validation

BASEDIR = os.path.join("output", "tms", "pretrain")


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
    ):
        self.X = np.ascontiguousarray(np.vstack([U_X, V_X]))
        self.P = np.vstack([U_posteriors, V_posteriors])
        self.hx_order, self.hx_sorted = self._build_hx(self.X)

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

    def _build_hx(self, X: np.ndarray):
        hx = self._fp64_rows(X)
        hx_order = np.argsort(hx)
        hx_sorted = hx[hx_order]
        return hx_order, hx_sorted

    def predict_proba(self, S: np.ndarray) -> np.ndarray:
        Sc = np.ascontiguousarray(S)
        hs = self._fp64_rows(Sc)
        left = np.searchsorted(self.hx_sorted, hs, side="left")
        right = np.searchsorted(self.hx_sorted, hs, side="right")
        idx_list = _lookup_sample(self.X, Sc, self.hx_order, left, right)
        return self.P[idx_list, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        posteriors = self.predict_proba(X)
        return posteriors.argmax(axis=-1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)


@dataclass
class ClassifierInfo:
    class_name: str
    params: dict
    default: bool
    ms_ignore: bool = False

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


@dataclass
class DatasetInfo:
    name: str
    collection: str
    n_classes: int


class DatasetBundle:
    def __init__(self, L_prevalence: np.ndarray, V: LabelledCollection, U: LabelledCollection):
        self.L_prevalence = L_prevalence
        self.V = V
        self.U = U
        self.test_prot = UPP(
            U,
            repeats=env.NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )
        self.V1, self.V2_prot = split_validation(self.V, random_state=qp.environ["_R_SEED"])

    def get_posteriors(self, h: PreTrainedClassifier):
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
            self.test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))

        return self


@dataclass
class PretainInfo:
    domain: str
    d_info: DatasetInfo
    h_info: ClassifierInfo

    def _get_stem(self):
        return f"{self.h_info.full_name}_{self.d_info.name}_{self.d_info.n_classes}"

    @property
    def info_path(self):
        stem = self._get_stem()
        return os.path.join(BASEDIR, self.domain, f"{stem}_info.pkl")

    @property
    def posteriors_path(self):
        stem = self._get_stem()
        return os.path.join(BASEDIR, self.domain, f"{stem}_post.npz")

    @property
    def exists(self) -> bool:
        return os.path.exists(self.info_path)

    def dump(self, V_posteriors: np.ndarray, U_posteriors: np.ndarray):
        os.makedirs(os.path.dirname(self.info_path), exist_ok=True)
        with open(self.info_path, "wb") as f:
            pickle.dump(self, f)
        np.savez_compressed(self.posteriors_path, V_posteriors=V_posteriors, U_posteriors=U_posteriors)

    @classmethod
    def load(cls, path: str, fast: bool = False) -> Tuple[DatasetBundle, PreTrainedClassifier, Self] | Self:
        with open(path, "rb") as f:
            p_info = pickle.load(f)
        if fast:
            return p_info

        L_prevalence, V, U = load_from_collection(p_info)

        post_path = p_info.posteriors_path
        _npz = np.load(post_path)
        V_posteriors = _npz["V_posteriors"]
        U_posteriors = _npz["U_posteriors"]
        h = PreTrainedClassifier(U_X=U.X, U_posteriors=U_posteriors, V_X=V.X, V_posteriors=V_posteriors)
        d_bundle = DatasetBundle(L_prevalence, V, U)

        return d_bundle, h, p_info


def load_from_collection(p_info: PretainInfo):
    dataset_collection = p_info.d_info.collection
    dataset_name = p_info.d_info.name
    if dataset_collection == "uci_binary":
        L, V, U = fetch_UCIBinaryDataset(dataset_name)
        return L.prevalence(), V, U
    elif dataset_collection == "uci_multiclass":
        L, V, U = fetch_UCIMulticlassDataset(dataset_name)
        return L.prevalence(), V, U
    else:
        raise ValueError(f"Unknown dataset collection: {dataset_collection}")


def load_info_paths(domain: str | None = None, problem: Literal["binary", "multiclass"] | None = None):
    domain = "*" if domain is None else domain
    _dir = os.path.join(BASEDIR, domain)
    paths = glob(os.path.join(_dir, "*_info.pkl"))
    if problem == "binary":
        paths = [p for p in paths if "_2_info.pkl" in p]
    elif problem == "multiclass":
        paths = [p for p in paths if "_2_info.pkl" not in p]

    return paths


@dataclass
class ClassifierDatasetBundle:
    dataset_name: str
    dataset_collection: str
    n_classes: int
    h_class_name: str
    h_params: dict
    h_default: bool
    h_ms_ignore: bool

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as f:
            return pickle.load(f)
