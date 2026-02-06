from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import quapy as qp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cap.utils.commons import parallel
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import KFold

from data import NotPretrainedError, PretainInfo, PreTrainedClassifier, load_info_paths
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
    posteriors: dict = None
    logits: dict = None
    exists: bool = False
    ece_pre: float = 0.0
    ece_post: float = 0.0


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
    def __init__(self, n_splits=5, lr=0.01, max_iter=50):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.n_splits = n_splits
        self.lr = lr
        self.max_iter = max_iter

    def forward(self, logits):
        return logits / self.temperature

    def _fit_fold(self, val_logits: torch.tensor, val_labels: torch.tensor):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            self.temperature.clamp_(min=0.01)

        optimal_T = self.temperature.item()
        return optimal_T

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        if self.n_splits <= 1:
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            val_logits_pt = torch.tensor(val_logits)
            val_labels_pt = torch.tensor(val_labels)
            T = self._fit_fold(val_logits_pt, val_labels_pt)
            val_probs = self.predict_proba(val_logits)
            return T, ece(val_labels, val_probs)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=qp.environ["_R_SEED"])

        temperatures = []
        ece_scores = []

        for train_index, test_index in kf.split(val_logits):
            train_logits = torch.tensor(val_logits[train_index])
            train_labels = torch.tensor(val_labels[train_index])
            test_logits = val_logits[test_index]
            test_labels = val_labels[test_index]

            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            self._fit_fold(train_logits, train_labels)
            temperatures.append(self.temperature.item())

            test_probs = self.predict_proba(test_logits)
            ece_scores.append(ece(test_labels, test_probs))

        mean_T = np.mean(temperatures)
        mean_ece = np.mean(ece_scores)

        self.temperature = nn.Parameter(torch.ones(1) * mean_T)

        return mean_T, mean_ece

    def predict_proba(self, test_logits: np.ndarray):
        _logits = torch.tensor(test_logits)
        with torch.no_grad():
            scaled_logits = self.forward(_logits)
            probs = F.softmax(scaled_logits, dim=1)

        return probs.cpu().numpy()


def calibrate(p: PretainInfo):
    d_bundle = p.load_dataset_bundle()
    V_logits, U_logits = p.load_logits()

    _exist = True
    try:
        p.load_posteriors()
    except NotPretrainedError:
        _exist = False

    if _exist:
        return Calibrated(p, exists=True)

    ece_pre = ece(d_bundle.V.y, F.softmax(V_logits, dim=1))
    ts = TemperatureScaling()
    _, ece_post = ts.fit(V_logits, d_bundle.V.y)

    logits = dict(V=V_logits, U=U_logits)
    posteriors = dict(V=ts.predict_proba(V_logits), U=ts.predict_proba(U_logits))

    return Calibrated(p=p, posteriors=posteriors, logits=logits, ece_pre=ece_pre, ece_post=ece_post)


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
            # c.p.dump(posteriors=c.posteriors, logits=c.logits)
            log.info(f"[{c.p.h_info.name}@{c.p.d_info.name}] calibrated: {c.ece_pre} -> {c.ece_post}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of jobs to use")
    pargs = parser.parse_args()

    main(pargs)
