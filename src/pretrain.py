from traceback import print_exception
from typing import Iterable, Tuple

import cap
from cap.utils.commons import parallel
from quapy.data import LabelledCollection

from config import gen_classifiers, gen_datasets
from data import ClassifierDatasetBundle
from env import PROJECT
from util import get_logger

EXPERIMENT = "pretrain"
log = get_logger(id=f"{PROJECT}.{EXPERIMENT}")


def train_variants(args: Tuple[str, str, LabelledCollection, LabelledCollection, LabelledCollection]):
    dataset_name, dataset_coll, L, V, U = args
    n_classes = L.n_classes

    bundles = []
    for clsf in gen_classifiers(n_classes):
        clsf.h.fit(*L.Xy)
        V_posteriors = clsf.h.predict_proba(V.X)
        U_posteriors = clsf.h.predict_proba(U.X)

        bundle = ClassifierDatasetBundle(
            dataset_name=dataset_name,
            dataset_collection=dataset_coll,
            n_classes=n_classes,
            V_posteriors=V_posteriors,
            U_posteriors=U_posteriors,
            h_class_name=clsf.class_name,
            h_params=clsf.params,
            h_default=clsf.default,
            h_ms_ignore=clsf.ms_ignore,
        )
        bundles.append(bundle)

    return bundles


def pretrain():
    datasets = []
    for dataset in gen_datasets():
        dataset_name, dataset_coll, (L, V, U) = dataset
        datasets.append((dataset_name, dataset_coll, L, V, U))

    bundles_gen: Iterable[Iterable[ClassifierDatasetBundle]] = parallel(
        func=train_variants,
        args_list=datasets,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for bundles in bundles_gen:
        for b in bundles:
            log.info(f"Pretrained {b.h_name} on {b.dataset_name}.")
            b.save()


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        pretrain()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
