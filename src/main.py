import os
from dataclasses import dataclass
from time import time
from traceback import print_exception
from typing import Iterable

import cap
import numpy as np
import quapy as qp
from cap.models.cont_table import LEAP
from cap.utils.commons import get_shift, parallel

import env
from config import (
    gen_acc_measure,
    gen_methods,
    get_acc_names,
    get_method_names,
)
from data import ClassifierInfo, PretainInfo, load_info_paths
from env import PROJECT
from method.base import ModelSelectionMethod, NeedsValidationProtocol
from results import RDF
from util import (
    all_results_exist,
    get_logger,
    get_plain_prev,
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
    p: PretainInfo
    acc_name: str
    method_name: str
    df: RDF = None
    t_train: float = None
    t_test_ave: float = None
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


def exp_protocol(args: tuple[str, str, ModelSelectionMethod]) -> EXP:
    # bundle_path, method_name, method, acc_name, acc_fn = args
    # clsf, D, method_name, method, val, val_posteriors = args
    info_path, method_name, method = args
    results = []

    D, h, p = PretainInfo.load(info_path)
    d_info, h_info = p.d_info, p.h_info
    D.get_posteriors(h)
    if isinstance(method, NeedsValidationProtocol):
        val, val_posteriors = D.V1, D.V1_posteriors
        method.set_validation_protocol(D.V2_prot, D.V2_prot_posteriors)
    else:
        val, val_posteriors = D.V, D.V_posteriors

    # fit MS method
    try:
        tinit = time()
        method.fit(val, val_posteriors, D.test_prot, D.test_prot_posteriors)
        t_train = time() - tinit
    except Exception as e:
        results.append(EXP.ERROR(e, p, "fit", method_name))
        return results

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    df_len = D.test_prot.total()
    test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
    # tp_true_cts = [ct.ravel() for ct in D.test_prot_true_cts]
    tp_true_cts = D.test_prot_true_cts

    for acc_name, acc_fn in gen_acc_measure(d_info.n_classes > 2):
        path = local_path(p.domain, d_info.name, h_info.full_name, method_name, acc_name, experiment=EXPERIMENT)
        if os.path.exists(path):
            results.append(EXP.EXISTS(p, acc_name, method_name))
            continue

        try:
            tinit = time()
            ranking_vals = method.rank(acc_fn)
            t_test_ave = (time() - tinit) / df_len
        except Exception as e:
            print_exception(e)
            results.append(EXP.ERROR(e, p, acc_name, method_name))
            continue

        # df_len = len(estim_accs)
        method_df = RDF.from_records(
            df_len,
            uids=np.arange(df_len).tolist(),
            shifts=test_shift,
            true_cts=tp_true_cts,
            ranking_vals=ranking_vals,
            classifier=h_info.name,
            classifier_class=h_info.class_name,
            default_c=[h_info.default] * df_len,
            ms_ignore=[h_info.ms_ignore] * df_len,
            method=method_name,
            dataset=d_info.name,
            collection=d_info.collection,
            n_classes=d_info.n_classes,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )

        results.append(
            EXP.SUCCESS(
                p,
                acc_name,
                method_name,
                df=method_df,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )

    return results


def experiments():
    experiment_args = []
    info_paths = load_info_paths(domain=env.DOMAIN)
    filtered_paths = []
    for path in info_paths:
        p = PretainInfo.load(path, fast=True)
        d_info, h_info = p.d_info, p.h_info
        if not all_results_exist(
            p.domain, d_info.name, h_info.full_name, get_method_names(), get_acc_names(), EXPERIMENT
        ):
            filtered_paths.append(path)
        else:
            log.info(f"[{h_info.name}@{d_info.name}] all results exist, skipping")

    for info_path in filtered_paths:
        for method_name, method in gen_methods():
            experiment_args.append((info_path, method_name, method))

    results_gen: Iterable[list[EXP]] = parallel(
        func=exp_protocol,
        args_list=experiment_args,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for res in results_gen:
        for r in res:
            if r.ok:
                path = local_path(
                    r.p.domain,
                    r.p.d_info.name,
                    r.p.h_info.full_name,
                    r.method_name,
                    r.acc_name,
                    experiment=EXPERIMENT,
                )
                r.df.save_result(path)
                log.info(
                    f"[{r.h_info.name}@{r.dataset_name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(f"[{r.h_info.name}@{r.dataset_name}] {r.method_name} on {r.acc_name} exists, skipping")
            elif r.error:
                log.warning(
                    f"[{r.h_info.name}@{r.dataset_name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
