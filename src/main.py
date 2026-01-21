import os
from dataclasses import dataclass
from traceback import print_exception
from typing import Iterable

import cap
import numpy as np
import pandas as pd
import quapy as qp
from cap.models.cont_table import LEAP
from cap.utils.commons import get_shift, parallel
from quapy.data import LabelledCollection

from config import (
    ClsVariant,
    DatasetBundle,
    gen_acc_measure,
    gen_classifiers,
    gen_datasets,
    gen_methods,
    get_acc_names,
    get_method_names,
)
from data import ClsfDataset
from env import PROJECT
from method.base import ModelSelection
from util import (
    all_exist_pre_check,
    gen_method_df,
    get_logger,
    get_plain_prev,
    is_excluded,
    local_path,
    timestamp,
)

EXPERIMENT = "main"
log = get_logger(id=f"{PROJECT}.{EXPERIMENT}")

qp.environ["SAMPLE_SIZE"] = 1000


class NoMSException(Exception):
    def __init__(self, message="Classifier does not support MS"):
        super().__init__(message)


def get_extra_from_method(df, method):
    if isinstance(method, LEAP):
        df["true_solve"] = method._true_solve_log[-1]


@dataclass
class EXP:
    code: int
    clsf: ClsVariant
    dataset_name: str
    acc_name: str
    method_name: str
    df: pd.DataFrame = None
    t_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


def exp_protocol(
    args: tuple[
        ClsVariant,
        DatasetBundle,
        str,
        ModelSelection,
        LabelledCollection,
        np.ndarray,
    ],
) -> list[EXP]:
    clsf, D, method_name, method, val, val_posteriors = args
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(clsf.name, D.dataset_name, method_name, acc_name):
            continue
        path = local_path(D.dataset_name, clsf.file_name, method_name, acc_name, experiment=EXPERIMENT)
        if os.path.exists(path):
            results.append(EXP.EXISTS(clsf, D.dataset_name, acc_name, method_name))
            continue

        df_len = D.test_prot.total()
        test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()

        try:
            ms_res = method.rank(acc_fn, val, val_posteriors)
        except Exception as e:
            print_exception(e)
            results.append(EXP.ERROR(e, clsf, D.dataset_name, acc_name, method_name))
            continue

        # df_len = len(estim_accs)
        method_df = gen_method_df(
            df_len,
            uids=np.arange(df_len).tolist(),
            shifts=test_shift,
            true_accs=D.true_accs[acc_name],
            classifier=clsf.name,
            classifier_class=clsf.class_name,
            default_c=[clsf.default] * df_len,
            ms_ignore=[clsf.ms_ignore] * df_len,
            method=method_name,
            dataset=D.dataset_name,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            **ms_res,
        )

        results.append(
            EXP.SUCCESS(
                clsf,
                D.dataset_name,
                acc_name,
                method_name,
                df=method_df,
                t_ave=ms_res.get("t_ave", None),
            )
        )

    return results


def train_cls(cd: ClsfDataset):
    #
    # check if all results for current combination already exist
    # if so, skip the combination
    if all_exist_pre_check(
        dataset_name=cd.D.dataset_name,
        cls_name=cd.clsf.file_name,
        method_names=get_method_names(),
        acc_names=get_acc_names(),
        experiment=EXPERIMENT,
    ):
        return cd.already_done()
    else:
        cd.load()

        if not cd.clsf.loaded:
            # fit model
            print(f"{cd.clsf.name}@{cd.D.dataset_name} not trained")
            cd.clsf.h.fit(*cd.D.L.Xy)
        if not cd.D.loaded:
            # create dataset bundle
            print(f"{cd.clsf.name}@{cd.D.dataset_name} not built")
            cd.D.get_posteriors(cd.clsf.h)

        cd.D.get_true_accs(list(gen_acc_measure()))

        cd.save()

        # store h-dataset combination
        return cd


def experiments():
    # cls_train_args = list(gen_model_dataset(gen_classifiers, gen_datasets))
    cls_train_args = []
    for dataset in gen_datasets():
        dataset_name, (L, V, U) = dataset
        for model in gen_classifiers(L.n_classes):
            cls_train_args.append(ClsfDataset(model, dataset_name, L, V, U))

    cls_dataset_gen = parallel(
        func=train_cls,
        args_list=cls_train_args,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
    )

    cls_dataset = []
    for cd in cls_dataset_gen:
        if cd.all_results_exist:
            log.info(f"All results for {cd.clsf.name} over {cd.D.dataset_name} exist, skipping")
        else:
            log.info(f"Trained {cd.clsf.name} over {cd.D.dataset_name}")
            cls_dataset.append((cd.clsf, cd.D))

    exp_prot_args_list = []
    for clsf, D in cls_dataset:
        for method_name, method, val, val_posteriors in gen_methods(clsf, D):
            exp_prot_args_list.append(
                (
                    clsf,
                    D,
                    method_name,
                    method,
                    val,
                    val_posteriors,
                )
            )

    results_gen: Iterable[list[EXP]] = parallel(
        func=exp_protocol,
        args_list=exp_prot_args_list,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for res in results_gen:
        for r in res:
            if r.ok:
                path = local_path(
                    r.dataset_name,
                    r.clsf.file_name,
                    r.method_name,
                    r.acc_name,
                    experiment=EXPERIMENT,
                )
                r.df.to_json(path)
                log.info(
                    f"[{r.clsf.name}@{r.dataset_name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_ave)}]"
                )
            elif r.old:
                log.info(f"[{r.clsf.name}@{r.dataset_name}] {r.method_name} on {r.acc_name} exists, skipping")
            elif r.error:
                log.warning(
                    f"[{r.clsf.name}@{r.dataset_name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
