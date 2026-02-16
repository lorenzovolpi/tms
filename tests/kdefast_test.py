import os
from argparse import ArgumentParser
from collections import defaultdict

import cap
from cap.error import vanilla_acc
from cap.models.direct import DoC
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from tqdm import tqdm

from method._kdeyml_cuda import KDEyMLCuda
from method._leap_cuda import OLEAPCuda

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(1)

from time import time

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import (
    UCI_MULTICLASS_DATASETS,
    fetch_UCIMulticlassDataset,
)
from sklearn.neural_network import MLPClassifier

from data import PretainInfo, load_info_paths

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


def leap(domain):
    dataset = {
        "text": "community-datasets__yahoo_answers_topics",
        "image": "cifar10",
    }[domain]
    ps = [PretainInfo.load(p, fast=True) for p in load_info_paths(domain=domain)]
    pf = [p for p in ps if dataset == p.d_info.name]
    for p in pf:
        db = p.load_dataset_bundle()
        h = p.load_pretrained_classifier(db)

        db.get_posteriors(h)

        doc = DoC(vanilla_acc, db.V2_prot, db.V2_prot_posteriors)
        leap = OLEAPCuda()

        tqdm.write(f"{p.h_info.name}@{p.d_info.name}")
        leap.fit(*db.V.Xy, db.V_posteriors)
        doc.fit(db.V1, db.V1_posteriors)

        leap_accs = np.array(
            leap.with_acc(vanilla_acc).batch_predict([Ui.X for Ui in db.test_prot()], db.test_prot_posteriors)
        )
        doc_accs = np.array(doc.batch_predict(db.test_prot, db.test_prot_posteriors))
        true_accs = np.array([vanilla_acc(ct) for ct in db.test_prot_true_cts])
        leap_mae = cap.error.mae(true_accs, leap_accs)
        doc_mae = cap.error.mae(true_accs, doc_accs)

        tqdm.write(f"leap={leap_mae:.6f}; doc={doc_mae:.6f}\n")


def deep(domain):
    dataset = {
        "text": "yahoo",
        "image": "cifar100",
    }[domain]
    ps = [PretainInfo.load(p, fast=True) for p in load_info_paths(domain=domain)]
    pf = [p for p in ps if dataset in p.d_info.name]
    for p in pf:
        db = p.load_dataset_bundle()
        h = p.load_pretrained_classifier(db)

        db.get_posteriors(h)

        q = KDEyMLCuda()

        tqdm.write(f"{p.h_info.name}@{p.d_info.name}")
        q.fit(*db.V.Xy)

        prevs = q.batch_quantify([Ui.X for Ui in db.test_prot()])
        prevs = [prevs[i] for i in range(prevs.shape[0])]
        accs = [qp.error.mae(prev, Ui.prevalence()) for prev, Ui in zip(prevs, db.test_prot())]

        tqdm.write(f"acc={np.mean(accs):.6f}\n")


def classic():
    res_bin = defaultdict(list)
    for d in UCI_MULTICLASS_DATASETS:
        L, U = fetch_UCIMulticlassDataset(d).train_test
        tinit = time()
        q_orig = KDEyML(MLPClassifier()).fit(L)
        orig_train_t = time() - tinit
        tinit = time()
        q = KDEyMLCuda().fit(*L.Xy)
        _train_t = time() - tinit

        prot = UPP(U, sample_size=1000, repeats=100, random_state=0, return_type="labelled_collection")

        q_orig_accs, q_accs = [], []
        orig_test_t, _test_t = [], []
        for Ui in prot():
            tinit = time()
            q_orig_pred = q_orig.quantify(Ui.X)
            orig_test_t.append(time() - tinit)
            q_orig_accs.append(qp.error.mae(Ui.prevalence(), q_orig_pred))

        tinit = time()
        q_preds = q.batch_quantify([Ui.X for Ui in prot()])
        _test_t = (time() - tinit) / prot.total()
        q_accs = [qp.error.mae(Ui.prevalence(), q_pred) for Ui, q_pred in zip(prot(), q_preds)]

        res_bin["dataset"].extend([d] * 2)
        res_bin["q"].append("orig")
        res_bin["acc"].append(np.mean(q_orig_accs))
        res_bin["train_t"].append(orig_train_t)
        res_bin["test_t"].append(np.mean(orig_test_t))
        res_bin["q"].append("fast")
        res_bin["acc"].append(np.mean(q_accs))
        res_bin["train_t"].append(_train_t)
        res_bin["test_t"].append(_test_t)
        print(
            f"{d} done. q_orig=[{orig_train_t:.3f}+{float(np.mean(orig_test_t)):.3f}s]; q=[{_train_t:.3f}+{float(np.mean(_test_t)):.3f}s]"
        )

    df = pd.DataFrame.from_dict(res_bin)
    pivot = pd.pivot_table(df, index=["dataset"], columns=["q"], values=["acc", "train_t", "test_t"])
    pivot.to_markdown("output/kdefast_mul.md")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    parser.add_argument("--leap", action="store_true")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.leap:
        leap(pargs.domain)
    elif pargs.domain == "classic":
        classic()
    elif pargs.domain in ["text", "image"]:
        deep(pargs.domain)
