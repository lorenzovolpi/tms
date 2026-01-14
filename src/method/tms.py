from time import time
from typing import override

import numpy as np
import quapy as qp
from cap.models.cont_table import O_LEAP
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from sklearn.neural_network import MLPClassifier

from data import ClsVariant, DatasetBundle
from method.base import ModelSelection


class TMS(ModelSelection): ...


class LEAP(TMS):
    @override
    def rank(self, acc_fn, val: LabelledCollection, val_posteriors: np.ndarray):
        if self.clsf.ms_ignore:
            return self.empty_rank()

        tinit = time()

        leap = O_LEAP(acc_fn, KDEyML(MLPClassifier())).fit(val, val_posteriors)
        ranking_vals = leap.batch_predict(self.D.test_prot, self.D.test_prot_posteriors, get_estim_cts=False)

        t_ave = (time() - tinit) / self.D.test_prot.total()

        return dict(
            ranking_vals=ranking_vals,
            t_ave=t_ave,
        )


class RQBS(TMS):
    """
    Reverse Quantification-Based Sampling
    """

    val_sidx_dict = {}

    def __init__(
        self,
        clsf: ClsVariant,
        D: DatasetBundle,
        n_vsamples=100,
        sample_size=qp.environ["SAMPLE_SIZE"],
    ):
        super().__init__(clsf, D)
        self.n_vsamples = n_vsamples
        self.sample_size = sample_size

    @override
    def rank(self, acc_fn, val: LabelledCollection, val_posteriors: np.ndarray):
        if self.clsf.ms_ignore:
            return self.empty_rank()

        tinit = time()

        sidx = self.val_sidx_dict.get((self.D.dataset_name, self.D.n_classes), None)
        if sidx is None:
            q = KDEyML(MLPClassifier()).fit(val)
            q_hats = [q.quantify(Ui.X) for Ui in self.D.test_prot()]
            sidx = np.asarray(
                [[val.sampling_index(self.sample_size, *q_hat) for _ in range(self.n_vsamples)] for q_hat in q_hats]
            )
            self.val_sidx_dict[(self.D.dataset_name, self.D.n_classes)] = sidx

        ranking_vals = []
        for vidxs in sidx:
            accs = []
            for idx in vidxs:
                vali_yhat = val_posteriors[idx, :].argmax(axis=1)
                vali_y = val[idx, :]
                accs.append(acc_fn(vali_yhat, vali_y))
            ranking_vals.append(np.median(accs))
        ranking_vals = np.array(ranking_vals)

        t_ave = (time() - tinit) / self.D.test_prot.total()

        return dict(
            ranking_vals=ranking_vals,
            t_ave=t_ave,
        )
