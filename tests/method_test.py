import itertools as IT

import numpy as np
from cap.data.datasets import fetch_UCIBinaryDataset
from sklearn.linear_model import LogisticRegression

from config import ClsVariant, DatasetBundle
from main import exp_protocol, train_cls
from method.tms import LEAP


def gen_classifiers(n_classes):
    cls_classes = [
        # 3 LR variants
        (
            "LR",
            LogisticRegression(),
            {
                "C": np.logspace(-1, 1, 3),
            },
        )
    ]

    for name, base, param_grid in cls_classes:
        _par_names = list(param_grid.keys())
        _par_combos = IT.product(*list(param_grid.values()))
        for _combo in _par_combos:
            _params = dict(zip(_par_names, _combo))
            yield ClsVariant(class_name=name, h=base, params=_params)


def gen_datasets():
    uci_binary = ["spambase"]
    for dn in uci_binary:
        yield dn, fetch_UCIBinaryDataset(dn)


def gen_methods(clsf: ClsVariant, D: DatasetBundle):
    yield "TMS_LEAP", LEAP(clsf, D), D.V, D.V_posteriors


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

    for res in results:
        for r in res:
            print(r.df)
