from argparse import ArgumentParser

import numpy as np
import quapy as qp
from cap.error import accuracy_score, vanilla_acc
from cap.utils.commons import contingency_table
from scipy.stats import chisquare, shapiro

from data import PretainInfo, load_info_paths

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


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
    r_o = np.full(n, 1 / n**0.5)
    k = np.argmin(p, axis=0)
    qb = np.zeros(n)
    qb[k] = 1
    print(f"{r=}, {np.linalg.norm(r)**2=}, {(r @ r_o) ** 2=}, {(p[k]-1)**2=}, {np.sum(p**2)=}, {p[k]**2=}")
    print((1 - ((r @ r_o) / np.linalg.norm(r)) ** 2) ** 0.5)
    print(np.linalg.norm(p - qb))
    print((p[k] - 1) ** 2 + np.sum(p**2) - p[k] ** 2)
    print((p - qb) @ r / (np.linalg.norm(r) * np.linalg.norm(p - qb)))
    print(np.linalg.norm(r_o))
    bias = ((np.linalg.norm(r) ** 2 - (r @ r_o) ** 2) * ((p[k] - 1) ** 2 + np.sum(p**2) - p[k] ** 2)) ** 0.5
    return float(bias)


def bounds(pargs):
    info_paths = load_info_paths(pargs.domain)

    for path in info_paths[:1]:
        db, h, pf = PretainInfo.load(path)
        val = db.V
        test = db.U
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
            bias = r @ (p - q)
            estim_acc = float(val_acc + bias)
            # print(f"{i}: \t{test_acc=}; {estim_acc=}")
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    bounds(pargs)
