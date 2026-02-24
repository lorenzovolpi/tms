from typing import Literal, Self

import numpy as np
import quapy as qp
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.functional import prevalence_from_labels
from quapy.method.aggregative import KDEyML
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from data import PretainInfo, load_info_paths

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


class rscore:
    def __init__(self, h, n_sample=1000, batch_size=10, quantify_q=True, verbose=False):
        self.h = h
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.quantify_q = quantify_q
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.classes_ = np.unique(y)
        self.n_classes = np.unique(y).shape[0]
        self.p = prevalence_from_labels(y, self.classes_)
        self.quant = KDEyML(MLPClassifier()).fit(LabelledCollection(X, y, self.classes_))
        if self.verbose:
            print("quantifier fit")
        y_hat = np.argmax(self.h.predict_proba(X), axis=-1)
        ct = contingency_table(y, y_hat, self.n_classes)
        self.r = np.diag(ct) / self.p

        biases = []
        rng = np.random.default_rng(seed=qp.environ["_R_SEED"])
        for i in tqdm(range(self.n_sample), desc="sampling", disable=not self.verbose):
            if self.quantify_q and i % self.batch_size == 0:
                Pi_idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
                Pi_X = X[Pi_idx]
                Pi_y = y[Pi_idx]
                Pi_y_hat = y_hat[Pi_idx]
                Pi_p = prevalence_from_labels(Pi_y, self.classes_)
                Pi_quant = KDEyML(MLPClassifier()).fit(LabelledCollection(Pi_X, Pi_y, self.classes_))
                cti = contingency_table(Pi_y, Pi_y_hat, self.n_classes)
                ri = np.diag(cti) / Pi_p
            else:
                Pi_idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
                Pi_y = y[Pi_idx]
                Pi_y_hat = y_hat[Pi_idx]
                Pi_p = prevalence_from_labels(Pi_y, self.classes_)
                cti = contingency_table(Pi_y, Pi_y_hat, self.n_classes)
                ri = np.diag(cti) / Pi_p

            Qi_idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            if self.quantify_q:
                Qi_X = X[Qi_idx]
                Qi_q_hat = Pi_quant.quantify(Qi_X)
            else:
                Qi_y = y[Qi_idx]
                Qi_q_hat = prevalence_from_labels(Qi_y, self.classes_)

            biases.append((Pi_p - Qi_q_hat) @ ri)

        self.biases = np.sort(np.abs(np.array(biases)))
        print(self.biases)
        return self

    def predict(self, X: np.ndarray) -> float:
        q_hat = self.quant.quantify(X)
        _bias = np.abs((self.p - q_hat) @ self.r)
        p_value = 1 - np.searchsorted(self.biases, _bias, side="left") / self.n_sample

        return _bias, p_value


class rpqscore:
    def __init__(self, h, n_sample=1000, batch_size=10, quantify_q=True, verbose=False):
        self.h = h
        self.n_sample = n_sample
        self.batch_size = batch_size
        self.quantify_q = quantify_q
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.classes_ = np.unique(y)
        self.n_classes = np.unique(y).shape[0]
        self.p = prevalence_from_labels(y, self.classes_)
        self.quant = KDEyML(MLPClassifier()).fit(LabelledCollection(X, y, self.classes_))
        if self.verbose:
            print("quantifier fit")
        y_hat = np.argmax(self.h.predict_proba(X), axis=-1)
        ct = contingency_table(y, y_hat, self.n_classes)
        self.r = np.diag(ct) / self.p

        biases = []
        rng = np.random.default_rng(seed=qp.environ["_R_SEED"])
        for i in tqdm(range(self.n_sample), desc="sampling", disable=not self.verbose):
            if self.quantify_q and i % self.batch_size == 0:
                Pi_idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
                Pi_X = X[Pi_idx]
                Pi_y = y[Pi_idx]
                Pi_y_hat = y_hat[Pi_idx]
                Pi_p = prevalence_from_labels(Pi_y, self.classes_)
                Pi_quant = KDEyML(MLPClassifier()).fit(LabelledCollection(Pi_X, Pi_y, self.classes_))
                cti = contingency_table(Pi_y, Pi_y_hat, self.n_classes)
                ri = np.diag(cti) / Pi_p
            else:
                Pi_idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
                Pi_y = y[Pi_idx]
                Pi_y_hat = y_hat[Pi_idx]
                Pi_p = prevalence_from_labels(Pi_y, self.classes_)
                cti = contingency_table(Pi_y, Pi_y_hat, self.n_classes)
                ri = np.diag(cti) / Pi_p

            Qi_idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            if self.quantify_q:
                Qi_X = X[Qi_idx]
                Qi_q_hat = Pi_quant.quantify(Qi_X)
            else:
                Qi_y = y[Qi_idx]
                Qi_q_hat = prevalence_from_labels(Qi_y, self.classes_)

            biases.append((Pi_p - Qi_q_hat) @ ri)

        self.biases = np.sort(np.abs(np.array(biases)))
        print(self.biases)
        return self

    def predict(self, X: np.ndarray) -> float:
        q_hat = self.quant.quantify(X)
        _bias = np.abs((self.p - q_hat) @ self.r)
        p_value = 1 - np.searchsorted(self.biases, _bias, side="left") / self.n_sample

        return _bias, p_value


def compute_score(domain, score: Literal["r_score", "rpq_score"]):

    info_paths = load_info_paths(domain=domain)
    datasets = ["cifar10"]
    classifiers = ["microsoft__resnet-50"]
    for path in info_paths:
        p = PretainInfo.load(path, fast=True)
        if p.d_info.name not in datasets or p.h_info.name not in classifiers:
            continue

        print(f"{p.h_info.name}@{p.d_info.name}:")
        D = p.load_dataset_bundle()
        h = p.load_pretrained_classifier(D)

        if score == "r_score":
            scorer = rscore(h, quantify_q=False, n_sample=1000, verbose=True).fit(*D.V.Xy)
        elif score == "rpq_score":
            scorer = rpqscore(h)

        for i, Ui in enumerate(D.test_prot()):
            i_bias, i_pval = scorer.predict(Ui.X)
            print(f"\t{i}: bias={i_bias}; p_value={i_pval}")
        print()


if __name__ == "__main__":
    compute_score("image", "r_score")
