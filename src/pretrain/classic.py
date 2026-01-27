import os
from dataclasses import dataclass
from traceback import print_exception
from typing import Iterable, Literal, Tuple

import cap
import numpy as np
from cap.utils.commons import parallel
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator

from config import gen_classifiers, gen_datasets
from data import ClassifierInfo, DatasetInfo, PretainInfo
from env import PROJECT
from util import get_logger

EXPERIMENT = "pretrain"
DOMAIN = "classic"
log = get_logger(id=f"{PROJECT}.{EXPERIMENT}.{DOMAIN}")

BATCH_SIZE = 8

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"


class TrainResult:
    def __init__(
        self,
        p_info: PretainInfo,
        V_posteriors: np.ndarray,
        U_posteriors: np.ndarray,
        status: Literal["ok", "old"],
    ):
        self.p_info = p_info
        self.V_posteriors = V_posteriors
        self.U_posteriors = U_posteriors
        self.status = status

    @classmethod
    def ok(cls, p_info: PretainInfo, V_posteriors: np.ndarray, U_posteriors: np.ndarray):
        return TrainResult(p_info, V_posteriors, U_posteriors, "ok")

    @classmethod
    def old(cls, p_info: PretainInfo):
        return TrainResult(p_info, None, None, "old")

    @property
    def is_ok(self):
        return self.status == "ok"

    @property
    def is_old(self):
        return self.status == "old"

    @property
    def posteriors(self) -> dict:
        return dict(
            V_posteriors=self.V_posteriors,
            U_posteriors=self.U_posteriors,
        )


@dataclass()
class Posteriors:
    V_posteriors: np.ndarray
    U_posteriors: np.ndarray

    @property
    def VU(self):
        return dict(
            V_posteriors=self.V_posteriors,
            U_posteriors=self.U_posteriors,
        )


def train_variants(
    args: Tuple[
        str,
        str,
        LabelledCollection,
        LabelledCollection,
        LabelledCollection,
        Iterable[Tuple[BaseEstimator, ClassifierInfo]],
    ],
) -> list[Tuple[PretainInfo, Posteriors]]:
    dataset_name, dataset_coll, L, V, U, h_batch = args
    n_classes = L.n_classes
    d_info = DatasetInfo(dataset_name, dataset_coll, n_classes)

    results = []
    for h, h_info in h_batch:
        p_info = PretainInfo(DOMAIN, d_info, h_info)

        h.fit(*L.Xy)
        V_posteriors = h.predict_proba(V.X)
        U_posteriors = h.predict_proba(U.X)
        results.append((p_info, Posteriors(V_posteriors, U_posteriors)))

    return results


def pretrain():
    datasets_classifiers = []
    for dataset in gen_datasets():
        dataset_name, dataset_coll, (L, V, U) = dataset
        n_classes = L.n_classes
        d_info = DatasetInfo(dataset_name, dataset_coll, n_classes)
        i = 0
        clsf_batches = []
        for clsf in gen_classifiers(n_classes):
            _, h_info = clsf
            if PretainInfo(DOMAIN, d_info, h_info).exists:
                log.info(f"Already exists: {h_info.name} on {d_info.name}, skipping.")
                continue
            if i % 8 == 0:
                clsf_batches.append([])
            clsf_batches[-1].append(clsf)
        for batch in clsf_batches:
            datasets_classifiers.append((dataset_name, dataset_coll, L, V, U, batch))

    results_gen: Iterable[list[Tuple[PretainInfo, Posteriors]]] = parallel(
        func=train_variants,
        args_list=datasets_classifiers,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for results in results_gen:
        for p_info, post in results:
            log.info(f"Pretrained {p_info.h_info.name} on {p_info.d_info.name}.")
            p_info.dump(**post.VU)


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        pretrain()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
