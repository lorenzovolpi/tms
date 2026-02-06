from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import quapy as qp
import torch
import torch.nn as nn
from cap.utils.commons import parallel
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import KFold

from data import PretainInfo, PreTrainedClassifier, load_info_paths
from env import PROJECT
from util import get_logger

EXPERIMENT = "calibrate"
DOMAIN = "image"

log = get_logger(id=f"{PROJECT}.{EXPERIMENT}.{DOMAIN}")
qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


@dataclass
class Calibrated:
    p: PretainInfo
    d: dict = None
    exists: bool = False
    pre_ece: float = 0.0
    post_ece: float = 0.0


def _ece_bin(y, post, n_bins=10):
    prob_true, prob_pred = calibration_curve(y, post, n_bins=n_bins)
    return np.mean(np.abs(prob_pred - prob_true))


def _ece_multi(y, post, n_bins=10):
    pass
    confidences = post.max(axis=1)
    predictions = post.argmax(axis=1)
    accuracies = (predictions == y).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            weight = mask.sum() / len(y)
            ece += weight * abs(bin_acc - bin_conf)

    return ece


def ece(y, post, n_bins=10):
    if post.shape[-1] == 2:
        return _ece_bin(y, post[:, 1], n_bins)
    else:
        return _ece_multi(y, post, n_bins)


# def calibrate(p: PretainInfo):
#     d_bundle = p.load_dataset_bundle()
#     _npz = np.load(p.posteriors_path)
#     if "calib_V_posteriors" in _npz and "calib_U_posteriors" in _npz:
#         return Calibrated(p, exists=True)
#
#     V_posteriors = _npz["V_posteriors"]
#     U_posteriors = _npz["U_posteriors"]
#
#     h = PreTrainedClassifier(U_X=d_bundle.U.X, U_posteriors=U_posteriors, V_X=d_bundle.V.X, V_posteriors=V_posteriors)
#     precal_V_post = h.predict_proba(d_bundle.V.X)
#     calib_h = CalibratedClassifierCV(h, method="temperature", ensemble=False).fit(*d_bundle.V.Xy)
#     postcal_V_post = calib_h.predict_proba(d_bundle.V.X)
#     postcal_U_post = calib_h.predict_proba(d_bundle.U.X)
#     pre_ece = ece(d_bundle.V.y, precal_V_post)
#     post_ece = ece(d_bundle.V.y, postcal_V_post)
#     post_dict = dict(
#         V_posteriors=V_posteriors,
#         U_posteriors=U_posteriors,
#         calib_V_posteriors=postcal_V_post,
#         calib_U_posteriors=postcal_U_post,
#     )
#     return Calibrated(p, post_dict, pre_ece=pre_ece, post_ece=post_ece)


class TemperatureScaling(nn.Module):
    def __init__(self, n_splits=5, lr=0.01):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.n_splits = n_splits
        self.lr = lr

    def forward(self, logits):
        return logits / self.temperature

    def _fit_fold(self):
        pass

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=qp.environ["_R_SEED"])

        temperatures = []
        ece_scores = []

        for train_index, test_index in kf.split(val_logits):
            train_logits = torch.tensor(val_logits[train_index])
            train_labels = torch.tensor(val_labels[train_index])
            test_logits = torch.tensor(val_logits[test_index])
            test_labels = val_labels[test_index]

            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            self._fit_fold(train_logits, train_labels)
            temperatures.append(self.temperature.item())

            test_probs = self.predict_proba(test_logits)
            ece_scores.append(ece(test_labels, test_probs))
        pass


def calibrate(p: PretainInfo):
    d_bundle = p.load_dataset_bundle()
    V_logits, U_logits = p.load_logits()


def main(pargs):
    info_paths = load_info_paths(DOMAIN)
    p_infos = [PretainInfo.load(p, fast=True) for p in info_paths]

    calib_gen: Iterable[Calibrated]
    if pargs.n_jobs > 1:
        calib_gen = parallel(
            func=calibrate,
            args_list=p_infos,
            n_jobs=8,
            return_as="generator_unordered",
            max_nbytes=None,
        )
    else:
        calib_gen = [calibrate(p) for p in p_infos]

    for c in calib_gen:
        if c.exists:
            log.info(f"[{c.p.h_info.name}@{c.p.d_info.name}] already calibrated, skipping.")
        else:
            # c.p.dump(**c.d)
            log.info(f"[{c.p.h_info.name}@{c.p.d_info.name}] calibrated.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of jobs to use")
    pargs = parser.parse_args()

    main(pargs)
