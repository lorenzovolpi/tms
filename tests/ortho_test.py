from argparse import ArgumentParser

import numpy as np
import quapy as qp
from cap.utils.commons import contingency_table
from scipy.stats import chisquare, shapiro

from data import PretainInfo, load_info_paths

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


def main(pargs):
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    main(pargs)
