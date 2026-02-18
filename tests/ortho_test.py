import itertools as IT
from argparse import ArgumentParser
from collections import defaultdict
from time import time
from typing import Callable, Tuple

import cap
import numpy as np
import pandas as pd
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset
from cap.error import accuracy_score, vanilla_acc
from cap.models import O_LEAP, DoC, direct
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from quapy.protocol import AbstractStochasticSeededProtocol
from scipy.stats import chisquare, shapiro
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from data import DatasetBundle, PretainInfo, load_info_paths

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


class BRAT(direct.CAPDirect):
    def __init__(self, acc_fn: Callable, protocol: AbstractStochasticSeededProtocol, prot_posteriors, clip_vals=(0, 1)):
        super().__init__(acc_fn)
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.clip_vals: Tuple[list, list] = clip_vals

    def sample_bias(self, X: np.ndarray) -> float:
        q_hat = self.quant.quantify(X)
        return (self.p - q_hat) @ self.r

    def _get_post_stats(self, sample: LabelledCollection, posteriors: np.ndarray):
        y_hat = np.argmax(posteriors, axis=-1)
        ct = contingency_table(sample.y, y_hat, sample.n_classes)
        acc = self.acc(ct)

        _bias = self.sample_bias(sample.X)

        return _bias, acc - self.val_acc

    def train_regression(self, prot_biases, prot_accs):
        docs = np.asarray(prot_biases).reshape(-1, 1)
        target = np.asarray(prot_accs)
        # reg = LinearRegression()
        reg = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.01))
        return reg.fit(docs, target)

    def predict_regression(self, test_bias):
        docs = np.asarray([test_bias]).reshape(-1, 1)
        pred_acc = self.reg_model.predict(docs)
        return pred_acc[0] + self.val_acc

    def fit(self, val: LabelledCollection, posteriors):
        self.p = val.prevalence()
        val_y_hat = np.argmax(posteriors, axis=-1)
        val_ct = contingency_table(val.y, val_y_hat, val.n_classes)
        self.val_acc = self.acc(val_ct)
        self.r = np.diag(val_ct) / self.p

        self.quant = KDEyML(MLPClassifier()).fit(val)

        prot_stats = [
            self._get_post_stats(sample, P) for sample, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]
        prot_biases, prot_accs = tuple(map(list, zip(*prot_stats)))

        self.reg_model = self.train_regression(prot_biases, prot_accs)

        return self

    def predict(self, X, posteriors):
        test_bias = self.sample_bias(X)
        acc_pred = self.predict_regression(test_bias)
        if self.clip_vals is not None:
            acc_pred = float(np.clip(acc_pred, *self.clip_vals))
        return acc_pred


def brat_test():
    h = LogisticRegression()

    res = defaultdict(list)
    for d in UCI_BINARY_DATASETS:
        L, V, U = fetch_UCIBinaryDataset(d)
        _h = clone(h).fit(*L.Xy)
        db = DatasetBundle(L.prevalence(), V, U).get_posteriors(_h)

        tinit = time()
        brat = BRAT(vanilla_acc, db.V2_prot, db.V2_prot_posteriors).fit(db.V1, db.V1_posteriors)
        brat_accs = np.array(brat.batch_predict(db.test_prot, db.test_prot_posteriors))
        t_brat = time() - tinit

        tinit = time()
        doc = DoC(vanilla_acc, db.V2_prot, db.V2_prot_posteriors).fit(db.V1, db.V1_posteriors)
        doc_accs = np.array(doc.batch_predict(db.test_prot, db.test_prot_posteriors))
        t_doc = time() - tinit

        leap = O_LEAP(vanilla_acc, KDEyML(MLPClassifier())).fit(db.V, db.V_posteriors)
        leap_accs = np.array(leap.batch_predict(db.test_prot, db.test_prot_posteriors))

        true_accs = np.array(list(map(vanilla_acc, db.test_prot_true_cts)))

        brat_mae = cap.error.mae(true_accs, brat_accs)
        doc_mae = cap.error.mae(true_accs, doc_accs)
        leap_mae = cap.error.mae(true_accs, leap_accs)

        res["brat"].append(brat_mae)
        res["doc"].append(doc_mae)
        res["leap"].append(leap_mae)
        res["dataset"].append(d)

        print(f"[{d}] brat={brat_mae:.6f}; doc={doc_mae:.6f}; {t_brat=:.3f}s; {t_doc=:.3f}s")

    df = pd.DataFrame.from_dict(res).set_index("dataset")
    print(df.to_string())


def ortho(pargs):
    info_paths = load_info_paths(pargs.domain)

    for path in info_paths:
        db, h, p = PretainInfo.load(path)
        for name, split in [("V", db.V)]:
            vp = h.predict_proba(split.X)
            v_ct = contingency_table(split.y, np.argmax(vp, axis=-1), split.n_classes)
            vr = np.diag(v_ct)
            v_acc = vr.sum() / v_ct.sum()
            mu = vr.mean()
            std = vr.std()
            _, chi2 = chisquare(f_obs=vr, f_exp=np.full(vr.shape, mu))
            _, p_value = shapiro(vr - mu)
            print(
                f"[{p.h_info.name}@{p.d_info.name}] {name}:\n\t{vr}\n\t{v_acc=:.4f}; {mu=:.4f}; {std=:.6f}; {chi2=:.4f}\n"
            )


def _bias_bound(r, p):
    n = p.shape[0]
    k = np.argmin(p, axis=0)
    qb = np.zeros(n)
    qb[k] = 1
    bias = (p - qb) @ r / np.linalg.norm(r)
    return float(bias)


def bounds(pargs):
    info_paths = load_info_paths(pargs.domain)

    for path in info_paths[:1]:
        db, h, pf = PretainInfo.load(path)
        val = db.V
        n = val.n_classes
        p = val.prevalence()
        y_hat = np.argmax(h.predict_proba(val.X), axis=-1)
        ct = contingency_table(val.y, y_hat, n)
        val_acc = vanilla_acc(ct)
        r = np.diag(ct) / p
        bias_bound = _bias_bound(r, p)

        print(f"[{pf.h_info.name}@{pf.d_info.name}] {bias_bound=}; {val_acc=}\n")
        ############ test knowledge #################
        for i, Ui in enumerate(db.test_prot()):
            test_acc = accuracy_score(Ui.y, np.argmax(h.predict_proba(Ui.X), axis=-1))
            q = Ui.prevalence()
            # bias = r @ (p - q)
            estim_acc = float(val_acc - bias_bound)
            print(f"{i}: \t{test_acc=}; {estim_acc=}")
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    brat_test()
