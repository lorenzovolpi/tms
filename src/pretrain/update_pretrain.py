import os
import pickle
from dataclasses import dataclass
from typing import Self, Tuple

import numpy as np
import quapy as qp
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.protocol import UPP

import env
from data import BASEDIR, ClassifierInfo, PreTrainedClassifier, load_from_collection
from util import split_validation


class DBundle:
    def __init__(
        self,
        name: str,
        collection: str,
        n_classes: int,
        L_prevalence: np.ndarray = None,
        V: LabelledCollection = None,
        U: LabelledCollection = None,
    ):
        self.name = name
        self.collection = collection
        self.n_classes = n_classes
        self.L_prevalence: np.ndarray = L_prevalence
        self.V = V
        if U:
            self.test_prot = UPP(
                U,
                repeats=env.NUM_TEST,
                return_type="labelled_collection",
                random_state=qp.environ["_R_SEED"],
            )
        if V:
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


def get_posteriors_path(dataset_name: str, n_classes: int, h_info: ClassifierInfo):
    return os.path.join(BASEDIR, f"{h_info.full_name}_{dataset_name}_{n_classes}_post.npz")


def old_load_info(
    info_path, fast=False
) -> Tuple[DBundle, PreTrainedClassifier, ClassifierInfo] | Tuple[DBundle, ClassifierInfo]:
    b = ClassifierDatasetBundle.load(info_path)
    h_info = ClassifierInfo(
        class_name=b.h_class_name,
        params=b.h_params,
        default=b.h_default,
        ms_ignore=b.h_ms_ignore,
    )
    if not fast:
        L, V, U = load_from_collection(b.dataset_collection, b.dataset_name)

        post_path = get_posteriors_path(b.dataset_name, b.n_classes, h_info)
        _npz = np.load(post_path)
        V_posteriors = _npz["V_posteriors"]
        U_posteriors = _npz["U_posteriors"]
        h = PreTrainedClassifier(U_X=U.X, U_posteriors=U_posteriors, V_X=V.X, V_posteriors=V_posteriors)
        dataset = DBundle(b.dataset_name, b.dataset_collection, b.n_classes, L.prevalence(), V, U)

        return dataset, h, h_info
    else:
        dataset = DBundle(b.dataset_name, b.dataset_collection, b.n_classes)
        return dataset, h_info
