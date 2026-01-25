import os
from traceback import print_exception
from typing import Iterable, Literal, Tuple

import cap
import numpy as np
from cap.utils.commons import parallel
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator

from config import gen_classifiers, gen_datasets
from data import BASEDIR, ClassifierDatasetBundle, ClassifierInfo, dump_info, get_info_path
from env import PROJECT
from util import get_logger

EXPERIMENT = "pretrain"
log = get_logger(id=f"{PROJECT}.{EXPERIMENT}")

BATCH_SIZE = 8


class TrainResult:
    def __init__(
        self,
        dataset_name: str,
        dataset_coll: str,
        n_classes: int,
        h_info: ClassifierInfo,
        V_posteriors: np.ndarray,
        U_posteriors: np.ndarray,
        status: Literal["ok", "old"],
    ):
        self.dataset_name = dataset_name
        self.dataset_coll = dataset_coll
        self.n_classes = n_classes
        self.h_info = h_info
        self.V_posteriors = V_posteriors
        self.U_posteriors = U_posteriors
        self.status = status

    @classmethod
    def ok(cls, dataset_name, dataset_coll, n_classes, h_info, V_posteriors, U_posteriors):
        return TrainResult(dataset_name, dataset_coll, n_classes, h_info, V_posteriors, U_posteriors, "ok")

    @classmethod
    def old(cls, dataset_name, dataset_coll, n_classes, h_info):
        return TrainResult(dataset_name, dataset_coll, n_classes, h_info, None, None, "old")

    @property
    def is_ok(self):
        return self.status == "ok"

    @property
    def is_old(self):
        return self.status == "old"

    @property
    def data(self) -> dict:
        return dict(
            dataset_name=self.dataset_name,
            dataset_collection=self.dataset_coll,
            n_classes=self.n_classes,
            h_info=self.h_info,
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
) -> list[TrainResult]:
    dataset_name, dataset_coll, L, V, U, h_batch = args
    n_classes = L.n_classes

    results = []
    for h, h_info in h_batch:
        if os.path.exists(get_info_path(dataset_name, n_classes, h_info)):
            results.append(TrainResult.old(dataset_name, dataset_coll, n_classes, h_info))
            continue

        h.fit(*L.Xy)
        V_posteriors = h.predict_proba(V.X)
        U_posteriors = h.predict_proba(U.X)
        results.append(TrainResult.ok(dataset_name, dataset_coll, n_classes, h_info, V_posteriors, U_posteriors))

    return results


def pretrain():
    datasets_classifiers = []
    for dataset in gen_datasets():
        dataset_name, dataset_coll, (L, V, U) = dataset
        n_classes = L.n_classes
        i = 0
        clsf_batches = []
        for clsf in gen_classifiers(n_classes):
            if i % 8 == 0:
                clsf_batches.append([])
            clsf_batches[-1].append(clsf)
        for batch in clsf_batches:
            datasets_classifiers.append((dataset_name, dataset_coll, L, V, U, batch))

    results_gen: Iterable[list[TrainResult]] = parallel(
        func=train_variants,
        args_list=datasets_classifiers,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for results in results_gen:
        for r in results:
            if r.is_old:
                log.info(f"Already exists: {r.h_info.name} on {r.dataset_name}, skipping.")
            elif r.is_ok:
                log.info(f"Pretrained {r.h_info.name} on {r.dataset_name}.")
                dump_info(**r.data)


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        pretrain()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
