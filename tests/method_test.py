import itertools as IT

import numpy as np
import pandas as pd
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from config import ClsVariant, DatasetBundle
from main import exp_protocol, train_cls
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
    uci_binary = ["spambase", "tictactoe"]
    for dn in uci_binary:
        yield dn, fetch_UCIBinaryDataset(dn)
    uci_multi = ["molecular", "nursery"]
    for dn in uci_multi:
        yield dn, fetch_UCIMulticlassDataset(dn)


def gen_methods(clsf: ClsVariant, D: DatasetBundle):
    yield "IMS", IMS(clsf, D), D.V, D.V_posteriors
    yield "TMS_LEAP", LEAP(clsf, D), D.V, D.V_posteriors
    yield "TMS_RQBS", RQBS(clsf, D), D.V, D.V_posteriors


if __name__ == "__main__":
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

    results = [exp_protocol(arg) for arg in exp_prot_args_list]

    dfs = [r.df for res in results for r in res]
    res = Results(pd.concat(dfs, axis=0)).model_selection()
    res = Results.concat(
        [
            res.default_classifier_ms(),
            res.method_ms("TMS_LEAP"),
            res.oracle_ms(),
        ]
    )

    pivot = pd.pivot_table(res.df, index=["dataset"], columns=["method"], values=["true_accs"])
    print(pivot)
