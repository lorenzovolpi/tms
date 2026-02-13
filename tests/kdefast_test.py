import os
from argparse import ArgumentParser
from collections import defaultdict
from traceback import print_exception

from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from tqdm import tqdm

from method._kdeyml_cuda import KDEyMLCuda

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(1)

from time import time

import numpy as np
import pandas as pd
import quapy as qp
from cap.error import vanilla_acc
from cap.models.cont_table import O_LEAP
from quapy.data.datasets import (
    UCI_BINARY_DATASETS,
    UCI_MULTICLASS_DATASETS,
    fetch_UCIBinaryDataset,
    fetch_UCIMulticlassDataset,
)
from sklearn.neural_network import MLPClassifier

from data import PretainInfo, load_info_paths
from method._kdey import KDEyMLFast

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


def text():
    ps = [PretainInfo.load(p, fast=True) for p in load_info_paths(domain="text")]
    pf = [p for p in ps if "yahoo" in p.d_info.name]
    for p in pf:
        db = p.load_dataset_bundle()
        h = p.load_pretrained_classifier(db)

        db.get_posteriors(h)

        q = KDEyMLCuda()

        tqdm.write(f"{p.h_info.name}@{p.d_info.name}")
        q.fit(*db.V.Xy)

        accs = []
        for Ui in tqdm(db.test_prot(), desc="Testing", total=db.test_prot.total()):
            prev = q.quantify(Ui.X)
            accs.append(qp.error.mae(prev, Ui.prevalence()))

        tqdm.write(f"acc={np.mean(accs):.6f}\n")


def image():
    ps = [PretainInfo.load(p, fast=True) for p in load_info_paths(domain="image")]
    pf = [p for p in ps if "cifar100" in p.d_info.name]
    for p in pf:
        db = p.load_dataset_bundle()
        h = p.load_pretrained_classifier(db)

        db.get_posteriors(h)

        q = KDEyMLCuda(kde_train_chunk_size=None)

        tqdm.write(f"{p.h_info.name}@{p.d_info.name}")
        q.fit(*db.V.Xy)

        accs = []
        for Ui in tqdm(db.test_prot(), desc="Testing", total=db.test_prot.total()):
            prev = q.quantify(Ui.X)
            accs.append(qp.error.mae(prev, Ui.prevalence()))

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
            tinit = time()
            try:
                q_pred = q.quantify(Ui.X)
            except Exception as e:
                print(f"dataset {d}: ", end="")
                raise e
            _test_t.append(time() - tinit)
            q_accs.append(qp.error.mae(Ui.prevalence(), q_pred))
            q_orig_accs.append(qp.error.mae(Ui.prevalence(), q_orig_pred))

        res_bin["dataset"].extend([d] * 2)
        res_bin["q"].append("orig")
        res_bin["acc"].append(np.mean(q_orig_accs))
        res_bin["train_t"].append(orig_train_t)
        res_bin["test_t"].append(np.mean(orig_test_t))
        res_bin["q"].append("fast")
        res_bin["acc"].append(np.mean(q_accs))
        res_bin["train_t"].append(_train_t)
        res_bin["test_t"].append(np.mean(_test_t))
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
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.domain == "classic":
        classic()
    if pargs.domain == "text":
        text()
    if pargs.domain == "image":
        image()
