from abc import abstractmethod
from time import time
from typing import override

import numpy as np
import quapy as qp
from cap.models.cont_table import O_LEAP
from cap.utils.commons import get_shift
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from sklearn.neural_network import MLPClassifier

from config import ClsVariant, DatasetBundle
from method.base import ModelSelection
from util import gen_method_df, get_plain_prev


class TMS(ModelSelection):
    def __init__(self, acc_fn, clsf: ClsVariant, D: DatasetBundle):
        self.acc_fn = acc_fn
        self.clsf = clsf
        self.D = D

    @abstractmethod
    def fit(self, val: LabelledCollection, val_posteriors: np.ndarray): ...

    @abstractmethod
    def predict(self): ...


class TMS_LEAP(TMS):
    @override
    def fit(self, val: LabelledCollection, val_posteriors: np.ndarray):
        self.val = val
        self.val_posteriors = val_posteriors

        tinit = time()
        self.leap = O_LEAP(self.acc_fn, KDEyML(MLPClassifier())).fit(val, val_posteriors)
        self.t_train = time() - tinit

        return self

    @override
    def predict(self):
        tinit = time()
        ranking_vals = self.leap.batch_predict(self.D.test_prot, self.D.test_prot_posteriors, get_estim_cts=False)
        t_test_ave = (time() - tinit) / self.D.test_prot.total()

        return dict(
            ranking_vals=ranking_vals,
            t_train=self.t_train,
            t_test_ave=t_test_ave,
        )


class TMS_RQBS(TMS):
    """
    Reverse Quantification-Based Sampling
    """

    val_q_dict = {}

    def __init__(
        self,
        acc_fn,
        clsf: ClsVariant,
        D: DatasetBundle,
        n_vsamples=100,
        sample_size=qp.environ["SAMPLE_SIZE"],
    ):
        super().__init__(acc_fn, clsf, D)
        self.n_vsamples = n_vsamples
        self.sample_size = sample_size

    @override
    def fit(self, val: LabelledCollection, val_posteriors: np.ndarray):
        self.val = val
        self.val_posteriors = val_posteriors
        q = self.val_sidx_dict.get((self.D.dataset_name, self.D.n_classes), None)
        if q is None:
            q = KDEyML(MLPClassifier()).fit(val)
            self.val_q_dict[(self.D.dataset_name, self.D.n_classes)] = q

        tinit = time()
        q_hats = [q.quantify(Ui.X) for Ui in self.D.test_prot()]
        self.sidx = np.asarray(
            [[val.sampling_index(self.sample_size, *q_hat) for _ in range(self.n_vsamples)] for q_hat in q_hats]
        )
        self.t_train = time() - tinit

        return self

    @override
    def predict(self):
        ranking_vals = []
        tinit = time()
        for vidxs in self.sidx:
            accs = []
            for idx in vidxs:
                vali_yhat = self.val_posteriors[idx, :].argmax(axis=1)
                vali_y = self.val[idx, :]
                accs.append(self.acc_fn(vali_yhat, vali_y))
            ranking_vals.append(np.median(accs))
        ranking_vals = np.array(ranking_vals)
        t_test_ave = (time() - tinit) / self.D.test_prot.total()

        return dict(
            ranking_vals=ranking_vals,
            t_train=self.t_train,
            t_test_ave=t_test_ave,
        )
