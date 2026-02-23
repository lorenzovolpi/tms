import os
from argparse import ArgumentParser
from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception
from typing import Iterable

import cap
import numpy as np
import quapy as qp
from cap.models.cont_table import LEAP
from cap.utils.commons import contingency_table, get_shift, parallel

import env
from config import (
    gen_acc_measure,
    gen_methods,
    get_acc_names,
    get_method_names,
    get_selection_acc,
)
from data import PretainInfo, load_info_paths
from env import PROJECT
from method.base import ModelSelectionMethod, NeedsValidationProtocol
from results import RDF, Results
from util import (
    all_results_exist,
    get_logger,
    get_plain_prev,
    local_path,
    timestamp,
)

qp.environ["SAMPLE_SIZE"] = 1000


def experiment(n: int):
    if n == 0:
        return "main"
    else:
        return "nscale"


class NoMSException(Exception):
    def __init__(self, message="Classifier does not support MS"):
        super().__init__(message)


def get_extra_from_method(df, method):
    if isinstance(method, LEAP):
        df["true_solve"] = method._true_solve_log[-1]


@dataclass
class EXP:
    code: int
    acc_name: str
    method_name: str
    p: PretainInfo = None
    iter: int = None
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


def exp_protocol(args: tuple[list[str], str, ModelSelectionMethod, int]) -> EXP:
    # bundle_path, method_name, method, acc_name, acc_fn = args
    # clsf, D, method_name, method, val, val_posteriors = args
    dataset_info_paths, method_name, method, nscale = args
    stem = f"nscale{nscale}"
    results = []

    p_infos = [PretainInfo.load(path, fast=True) for path in dataset_info_paths]

    domain = p_infos[0].domain
    d_info = p_infos[0].d_info

    all_exist = True
    for acc in get_acc_names():
        all_exist = all_exist and os.path.exists(
            local_path(domain, d_info.name, stem, method_name, acc, experiment=experiment(nscale))
        )
    if all_exist:
        results.append(EXP.EXISTS("all", method_name))
        return results

    D = p_infos[0].load_dataset_bundle()
    hp_map = {p.h_info.name: p for p in p_infos}
    needs_val = isinstance(method, NeedsValidationProtocol)
    val = D.V1 if needs_val else D.V

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    df_len = D.test_prot.total()
    test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()

    for acc in get_acc_names():
        path = local_path(domain, d_info.name, stem, method_name, acc, experiment=experiment(nscale))
        if os.path.exists(path):
            results.append(EXP.EXISTS(acc, method_name))
            continue

        method_dict = dict(
            uids=np.arange(df_len).tolist(),
            shifts=test_shift,
            true_cts=[],
            ranking_vals=[],
            classifier=[],
            classifier_class=[],
            default_c=[],
            ms_ignore=[False] * df_len,
            method=method_name,
            dataset=d_info.name,
            collection=d_info.collection,
            n_classes=d_info.n_classes,
            acc_name=acc,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=[],
            t_test_ave=[],
        )

        or_base_dir = os.path.join(env.root_dir, experiment(nscale - 1))
        best_hs = Results.load(
            basedir=or_base_dir, acc_name=acc, dataset=d_info.name, domain=domain, filter_methods=[method_name]
        ).best_classifiers()

        _error = None
        for i, Ui in enumerate(D.test_prot()):
            h_name = best_hs[i]
            h_info = hp_map[h_name].h_info
            h = hp_map[h_name].load_pretrained_classifier(D)
            val_posteriors = h.predict_proba(val.X)
            Ui_post = h.predict_proba(Ui.X)
            true_ct = contingency_table(Ui.y, np.argmax(Ui_post, axis=1), d_info.n_classes)
            if needs_val:
                method.set_validation_protocol(D.V2_prot, [h.predict_proba(V2i.X) for V2i in D.V2_prot()])

            try:
                tinit = time()
                method.fit_single(val, val_posteriors, Ui, Ui_post)
                t_train = time() - tinit
                tinit = time()
                ranking_val = method.rank(get_selection_acc(acc, d_info.n_classes > 2))
                t_test = time() - tinit
            except Exception as e:
                print_exception(e)
                _error = e
                break

            method_dict["true_cts"].append(true_ct)
            method_dict["ranking_vals"].append(ranking_val)
            method_dict["classifier"].append(h_name)
            method_dict["classifier_class"].append(h_info.class_name)
            method_dict["default_c"].append(h_info.default)
            method_dict["t_train"].append(t_train)
            method_dict["t_test_ave"].append(t_test)

        if _error is not None:
            results.append(EXP.ERROR(_error, acc, method_name, p=hp_map[h_name], iter=i))
            continue

        method_dict["t_train"] = sum(method_dict["t_train"])
        method_dict["t_test_ave"] = sum(method_dict["t_test_ave"]) / df_len
        method_df = RDF.from_records(df_len, **method_dict)

        results.append(
            EXP.SUCCESS(
                acc,
                method_name,
                df=method_df,
                t_train=method_dict["t_train"],
                t_test_ave=method_dict["t_test_ave"],
            )
        )

    return results


def experiments(log: Logger, nscale: int):
    experiment_args = []
    info_paths = load_info_paths(domain=env.DOMAIN)
    filtered_paths = []
    for path in info_paths:
        p = PretainInfo.load(path, fast=True)
        d_info, h_info = p.d_info, p.h_info
        if not all_results_exist(
            p.domain, d_info.name, h_info.full_name, get_method_names(), get_acc_names(), experiment(nscale)
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
                    experiment=experiment(nscale),
                )
                r.df.save_result(path)
                log.info(
                    f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name} on {r.acc_name} exists, skipping")
            elif r.error:
                log.warning(
                    f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scale", action="store_const", const="nscale", default=1, type=int, help="Nscale experiments")
    pargs = parser.parse_args()

    log = get_logger(id=f"{PROJECT}.{experiment(pargs.nscale)}")

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments(log, pargs.nscale)
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
