import itertools as IT
import os
from typing import Iterable

import cap
import numpy as np
import pandas as pd
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.utils.commons import parallel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import main
from config import ClassifierInfo, DatasetInfo
from data import ClsfDataset, _get_classifier
from main import EXP, exp_protocol, train_cls
from method.ims import IMS
from method.tms import LEAP, RQBS
from results import RDF, Results

qp.environ["SAMPLE_SIZE"] = 1000
main.EXPERIMENT = "tests"


def gen_classifiers(n_classes):
    cls_classes = [
        # 3 LR variants
        (
            "LR",
            LogisticRegression(),
            {
                "C": np.logspace(-1, 1, 3),
            },
        ),
        (
            "MLP",
            MLPClassifier(),
            {
                "alpha": np.around(np.logspace(-3, -1, 3), decimals=5),
            },
        ),
    ]

    for name, base, param_grid in cls_classes:
        _par_names = list(param_grid.keys())
        _par_combos = IT.product(*list(param_grid.values()))
        for _combo in _par_combos:
            _params = dict(zip(_par_names, _combo))
            h = _get_classifier(base, _params)
            yield h, ClassifierInfo(class_name=name, h=base, params=_params)


def gen_datasets():
    uci_binary = [
        "spambase",
        "tictactoe",
    ]
    for dn in uci_binary:
        yield dn, fetch_UCIBinaryDataset(dn)
    uci_multi = [
        "molecular",
        "nursery",
    ]
    for dn in uci_multi:
        yield dn, fetch_UCIMulticlassDataset(dn)


def gen_methods(clsf: ClassifierInfo, D: DatasetInfo):
    # yield "IMS", IMS(clsf, D), D.V, D.V_posteriors
    yield "TMS_LEAP", LEAP(clsf, D), D.V, D.V_posteriors
    yield "TMS_RQBS", RQBS(clsf, D), D.V, D.V_posteriors


def get_method_names():
    mock_clsf = ClassifierInfo.mock()
    mock_D = DatasetInfo.mock()
    return [m for m, _, _, _ in gen_methods(mock_clsf, mock_D)]


if __name__ == "__main__":
    outdir = os.path.join("output", "tests")
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "results.json")
    ClsfDataset.BASE_PATH = ["output", "tests", "models"]

    if not os.path.exists(out_json):
        cls_train_args = []
        for dataset in gen_datasets():
            dataset_name, (L, V, U) = dataset
            for model in gen_classifiers(L.n_classes):
                cls_train_args.append(ClsfDataset(model, dataset_name, L, V, U))

        cls_dataset = [train_cls(arg) for arg in cls_train_args]
        cls_dataset = [cd for cd in cls_dataset if not cd.all_results_exist]
        for cd in cls_dataset:
            print(f"trained {cd.clsf.name}@{cd.D.dataset_name}")

        exp_prot_args_list = []
        for cd in cls_dataset:
            for method_name, method, val, val_posteriors in gen_methods(cd.clsf, cd.D):
                exp_prot_args_list.append((cd.clsf, cd.D, method_name, method, val, val_posteriors))

        # results = [exp_protocol(arg) for arg in exp_prot_args_list]
        results_gen: Iterable[list[EXP]] = parallel(
            func=exp_protocol,
            args_list=exp_prot_args_list,
            n_jobs=10,
            return_as="generator_unordered",
            max_nbytes=None,
        )

        results = []
        for res in results_gen:
            for r in res:
                if r.ok:
                    results.append(r)
                elif r.error:
                    print(f"{r.h_info.name}@{r.dataset_name}:{r.err}")

        dfs = [r.df for r in results]
        res = Results(pd.concat(dfs, axis=0))
        res.df.index = range(len(res.df))
        res.df.to_json(out_json)

    res = Results(RDF.load_result(out_json))
    res = Results.concat(
        [
            res.oracle_ms(),
            res.default_classifier_ms(),
        ]
        + [res.method_ms(m) for m in get_method_names()],
    )

    pivot = pd.pivot_table(res.df, index=["dataset"], columns=["method"], values=["true_accs"])
    print(pivot)
