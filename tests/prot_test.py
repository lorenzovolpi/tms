import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quapy as qp
import seaborn as sns
from quapy.functional import uniform_simplex_sampling
from tqdm import tqdm

import env

qp.environ["_R_SEED"] = 0


def dirichlet_prot(n_classes, size):
    rng = np.random.default_rng(seed=qp.environ["_R_SEED"])
    return rng.dirichlet(np.ones(n_classes), size=size)


def kraemer_prot(n_classes, size):
    return uniform_simplex_sampling(n_classes, size=size)


def shift_from_center(prevs: np.ndarray) -> np.ndarray:
    n_classes = prevs.shape[1]
    p = np.full((1, n_classes), 1 / n_classes)

    return np.linalg.norm(prevs - p, axis=1) * np.sqrt(n_classes / (n_classes - 1))


def main():
    n_samples = 20000
    _classes = np.logspace(0.5, 4, 10).astype(int)
    plot_dir = os.path.join(env.root_dir, "tests", "prot")
    os.makedirs(plot_dir, exist_ok=True)

    for nc in tqdm(_classes):
        dir_prevs = dirichlet_prot(nc, size=n_samples)
        kra_prevs = kraemer_prot(nc, size=n_samples)
        dir_shifts = shift_from_center(dir_prevs)
        kra_shifts = shift_from_center(kra_prevs)

        data = []
        for i in range(n_samples):
            data.append(dict(prot="dirichlet", shift=dir_shifts[i]))
            data.append(dict(prot="kraemer", shift=kra_shifts[i]))

        df = pd.DataFrame(data)
        # df["binned_shift"] = pd.cut(df["shift"], 50)

        sns.histplot(
            df,
            x="shift",
            hue="prot",
        )
        plt.savefig(os.path.join(plot_dir, f"{nc}.png"), dpi=300)
        plt.clf()


if __name__ == "__main__":
    main()
