import os
from collections import defaultdict
from traceback import print_exception

from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(1)

from time import time

import numpy as np
import pandas as pd
import quapy as qp
from cap.error import vanilla_acc
from cap.models.cont_table import O_LEAP
from quapy.data.datasets import UCI_BINARY_DATASETS, fetch_UCIBinaryDataset
from sklearn.neural_network import MLPClassifier

from data import PretainInfo, load_info_paths
from method._kdey import KDEyMLFast

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


def text():
    ps = [PretainInfo.load(p, fast=True) for p in load_info_paths(domain="text")]
    pf = [p for p in ps if "yelp_polarity" in p.d_info.name]
    for p in pf:
        db = p.load_dataset_bundle()
        h = p.load_pretrained_classifier(db)

        db.get_posteriors(h)

        leap = O_LEAP(vanilla_acc, KDEyMLFast(MLPClassifier(), limit_kde_train=1e4))

        tinit = time()
        leap.fit(db.V, db.V_posteriors)
        tfit = time() - tinit

        tinit = time()
        leap._batch_predict_ct(db.test_prot, db.test_prot_posteriors)
        ttest = time() - tinit

        print(f"[{p.h_info.name}@{p.d_info.name}] {tfit=}; {ttest=}")


def classic():
    res_bin = defaultdict(list)
    for d in UCI_BINARY_DATASETS:
        L, U = fetch_UCIBinaryDataset(d).train_test
        tinit = time()
        q_orig = KDEyML(MLPClassifier()).fit(L)
        orig_train_t = time() - tinit
        tinit = time()
        q = KDEyMLFast(MLPClassifier(), limit_kde_train=0.1).fit(L)
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
    pivot.to_markdown("output/kdefast_bin.md")


if __name__ == "__main__":
    classic()
