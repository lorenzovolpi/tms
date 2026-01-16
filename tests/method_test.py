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

from config import ClsVariant, DatasetBundle
from main import EXP, exp_protocol, train_cls
from method.ims import IMS
from method.tms import LEAP, RQBS
from results import Results

qp.environ["SAMPLE_SIZE"] = 1000


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
            yield ClsVariant(class_name=name, h=base, params=_params)


def gen_datasets():
    uci_binary = [
        "spambase",
        # "tictactoe",
    ]
    for dn in uci_binary:
        yield dn, fetch_UCIBinaryDataset(dn)
    uci_multi = [
        "molecular",
        # "nursery",
    ]
    for dn in uci_multi:
        yield dn, fetch_UCIMulticlassDataset(dn)


def gen_methods(clsf: ClsVariant, D: DatasetBundle):
    # yield "IMS", IMS(clsf, D), D.V, D.V_posteriors
    yield "TMS_LEAP", LEAP(clsf, D), D.V, D.V_posteriors
    yield "TMS_RQBS", RQBS(clsf, D), D.V, D.V_posteriors


def get_method_names():
    mock_clsf = ClsVariant.mock()
    mock_D = DatasetBundle.mock()
    return [m for m, _, _, _ in gen_methods(mock_clsf, mock_D)]


if __name__ == "__main__":
    outdir = os.path.join("output", "tests")
    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "results.json")

    if os.path.exists(out_json):
        res = Results(pd.read_json(out_json))
    else:
        cls_train_args = []
        for dataset in gen_datasets():
            _, (L, _, _) = dataset
            for model in gen_classifiers(L.n_classes):
                cls_train_args.append((model, dataset))

        cls_dataset = [train_cls(arg) for arg in cls_train_args]
        cls_dataset = [cd for cd in cls_dataset if not cd[1].empty]
        for clsf, D in cls_dataset:
            print(f"trained {clsf.name}@{D.dataset_name}")

        exp_prot_args_list = []
        for clsf, D in cls_dataset:
            for method_name, method, val, val_posteriors in gen_methods(clsf, D):
                exp_prot_args_list.append((clsf, D, method_name, method, val, val_posteriors))

        # results = [exp_protocol(arg) for arg in exp_prot_args_list]
        results: Iterable[list[EXP]] = parallel(
            func=exp_protocol,
            args_list=exp_prot_args_list,
            n_jobs=10,
            return_as="generator_unordered",
            max_nbytes=None,
        )

        dfs = [r.df for res in results for r in res]
        res = Results(pd.concat(dfs, axis=0))
        res.df.to_json(out_json)

    res = Results.concat(
        [
            res.oracle_ms(),
            res.default_classifier_ms(),
        ]
        + [res.method_ms(m) for m in get_method_names()],
    )

    pivot = pd.pivot_table(res.df, index=["dataset"], columns=["method"], values=["true_accs"])
    print(pivot)
