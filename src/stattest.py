import os
from argparse import ArgumentParser

os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = str(1)
from typing import Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quapy as qp
import seaborn as sns
from cap.utils.commons import contingency_table
from joblib import Parallel, delayed
from quapy.data import LabelledCollection
from quapy.functional import prevalence_from_labels
from quapy.method.aggregative import PACC, KDEyML
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

import env
from data import PretainInfo, load_info_paths

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
    "stanfordnlp__imdb": "imdb",
    "fancyzhx__yelp_polarity": "yelp-polarity",
    "stanfordnlp__sst2": "sst2",
    "fancyzhx__ag_news": "ag-news",
    "fancyzhx__dbpedia_14": "dbpedia-14",
    "community-datasets__yahoo_answers_topics": "yahoo-answers-topics",
    "yelp_reviews": "yelp-reviews",
    "ag_news-lt": "ag-news-lt",
    "dbpedia_14-lt": "dbpedia-14-lt",
    "ethz__food101": "food101",
}

clsf_map = {
    "microsoft__resnet-50": "resnet-50",
    "facebook__convnext-tiny-224": "convnext-tiny",
    "google__efficientnet-b0": "efficientnet-b0",
    "google__vit-base-patch16-224": "vit-base",
    "microsoft__swin-tiny-patch4-window7-224": "swin-tiny",
    "google-bert__bert-base-uncased": "bert",
    "FacebookAI__roberta-base": "roberta",
    "distilbert__distilbert-base-uncased": "distilbert",
    "microsoft__deberta-v3-base": "deberta-v3",
    "google__electra-base-discriminator": "electra",
}


class rscore:
    def __init__(self, h, n_sample=1000, batch_size=10, verbose=False, n_jobs=64):
        self.h = h
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.classes_ = np.unique(y)
        self.n_classes = np.unique(y).shape[0]
        self.p = prevalence_from_labels(y, self.classes_)
        self.quant = PACC(MLPClassifier()).fit(LabelledCollection(X, y, self.classes_))
        y_hat = np.argmax(self.h.predict_proba(X), axis=-1)
        ct = contingency_table(y, y_hat, self.n_classes)
        self.r = np.diag(ct) / self.p

        def compute_perm(perm):
            Xi, yi, yi_hat = X[perm], y[perm], y_hat[perm]

            _cut = len(X) - 1000
            Pi_X, Pi_y, Pi_y_hat = Xi[:_cut], yi[:_cut], yi_hat[:_cut]
            Pi_p = prevalence_from_labels(Pi_y, self.classes_)
            Pi_quant = PACC(MLPClassifier()).fit(LabelledCollection(Pi_X, Pi_y, self.classes_))
            cti = contingency_table(Pi_y, Pi_y_hat, self.n_classes)
            ri = np.diag(cti) / Pi_p

            Qi_X = Xi[_cut:]
            Qi_q_hat = Pi_quant.quantify(Qi_X)
            return (Pi_p - Qi_q_hat) @ ri

        rng = np.random.default_rng(seed=qp.environ["_R_SEED"])
        idx_perms = [rng.permutation(len(X)) for _ in range(self.n_sample)]

        # biases = [compute_perm(perm) for perm in idx_perms]
        biases_gen = Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
            delayed(compute_perm)(perm) for perm in idx_perms
        )

        biases = [
            _b
            for _b in tqdm(
                biases_gen, total=self.n_sample, desc=f"{self.__class__.__name__} fit", disable=not self.verbose
            )
        ]

        self.biases = np.sort(np.abs(np.array(biases)))
        print(self.biases)

        return self

    def score(self, X: np.ndarray) -> tuple[float, float]:
        q_hat = self.quant.quantify(X)
        _bias = np.abs((self.p - q_hat) @ self.r)
        p_value = 1 - np.searchsorted(self.biases, _bias, side="left") / self.n_sample

        return _bias, p_value


class rqscore:
    def __init__(self, h, n_sample=1000):
        self.h = h
        self.n_sample = n_sample

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.classes_ = np.unique(y)
        self.n_classes = np.unique(y).shape[0]
        self.quant = PACC(MLPClassifier()).fit(LabelledCollection(X, y, self.classes_))

        self.p = prevalence_from_labels(y, self.classes_)

        y_hat = np.argmax(self.h.predict_proba(X), axis=-1)
        ct = contingency_table(y, y_hat, self.n_classes)
        self.r = np.diag(ct) / self.p

        rng = np.random.default_rng(seed=qp.environ["_R_SEED"])
        qs = rng.dirichlet(np.ones(self.n_classes), size=self.n_sample)
        diffs = self.p.reshape(1, -1) - qs
        rs = rng.random((self.n_sample, self.n_classes))
        self.biases = np.sort(np.abs((diffs @ rs.T).flatten()))

        return self

    def score(self, X) -> tuple[float, float]:
        q_hat = self.quant.quantify(X)
        _bias = np.abs((self.p - q_hat) @ self.r)
        p_value = 1 - np.searchsorted(self.biases, _bias, side="left") / self.n_sample

        return _bias, p_value


def local_path(score, domain, d_name, h_name):
    res_dir = os.path.join(env.root_dir, "tms", "stattest", score, domain)
    os.makedirs(res_dir, exist_ok=True)
    return os.path.join(res_dir, f"{h_name}__{d_name}.parquet")


def compute_score(domain, score: Literal["rscore", "rqscore"], plot=False):

    info_paths = load_info_paths(domain=domain)
    datasets = ["cifar10"]
    results = []
    for path in info_paths:
        p = PretainInfo.load(path, fast=True)
        d_name = dataset_map.get(p.d_info.name, p.d_info.name)
        h_name = clsf_map.get(p.h_info.name, p.h_info.name)
        if p.d_info.name not in datasets:
            continue

        dest_path = local_path(score, domain, d_name, h_name)
        if os.path.exists(dest_path):
            _df = pd.read_parquet(dest_path)
            results.append(_df)
            print("loaded.\n")

        print(f"{p.h_info.name}@{p.d_info.name}:")
        D = p.load_dataset_bundle()
        h = p.load_pretrained_classifier(D)

        if score == "rscore":
            scorer = rscore(h, n_sample=1000, verbose=True).fit(*D.V.Xy)
        elif score == "rqscore":
            scorer = rqscore(h, n_sample=1000).fit(*D.V.Xy)

        data = []
        for i, Ui in tqdm(
            enumerate(D.test_prot()), total=D.test_prot.total(), desc=f"{scorer.__class__.__name__} scoring"
        ):
            i_bias, i_pval = scorer.score(Ui.X)
            data.append(
                dict(
                    dataset=d_name,
                    classifier=h_name,
                    bias=i_bias,
                    p_value=i_pval,
                )
            )

        _df = pd.DataFrame(data)
        _df.to_parquet(dest_path)

        results.append(_df)

        print()

    df = pd.concat(results)
    df["reject"] = df["p_value"] < 0.05
    pivot = pd.pivot_table(df, index=["dataset"], columns=["classifier"], values=["reject"])
    print(pivot.to_string())

    if plot:
        plot_dir = os.path.join(env.root_dir, "tms", "plots", "stattest")
        os.makedirs(plot_dir, exist_ok=True)
        for d_name in df["dataset"].unique():
            ddf = df.loc[df["dataset"] == d_name, :]
            plot = sns.histplot(ddf, x="p_value", hue="classifier", kde=True)
            plot.set_xlim(0, 1)
            plt.savefig(os.path.join(plot_dir, f"{score}_{domain}_{d_name}.png"), dpi=300)
            plt.clf()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--score", action="store", dest="score", choices=["rscore", "rqscore"])
    parser.add_argument("-d", "--domain", action="store", choices=["text", "image"])
    parser.add_argument("--plot", action="store_true")
    pargs = parser.parse_args()

    compute_score(pargs.domain, pargs.score, plot=pargs.plot)
