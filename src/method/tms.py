from abc import abstractmethod
from time import time
from typing import Callable, Literal, override

import cap
import cap.models.cont_table as cont_table
import cap.models.direct as direct
import numpy as np
import quapy as qp
from cap.models.base import CAP, ClassifierAccuracyPrediction
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from sklearn.neural_network import MLPClassifier

from data import ClsVariant, DatasetBundle
from method.base import ModelSelection


class TMS(ModelSelection): ...


class TMS_CAP(TMS):
    @override
    def rank(self, acc_fn: Callable, val: LabelledCollection, val_posteriors: np.ndarray):
        if self.clsf.ms_ignore:
            return self.empty_rank()

        tinit = time()

        model = self.get_cap_model(acc_fn, val, val_posteriors)
        ranking_vals = self.get_ranking_vals(model)

        t_ave = (time() - tinit) / self.D.test_prot.total()

        return dict(
            ranking_vals=ranking_vals,
            t_ave=t_ave,
        )

    @abstractmethod
    def get_cap_model(self, acc_fn: Callable, val: LabelledCollection, val_posteriors: np.ndarray): ...

    def get_ranking_vals(self, model: CAP):
        return model.batch_predict(self.D.test_prot, self.D.test_prot_posteriors)


class LEAP(TMS_CAP):
    @override
    def get_cap_model(self, acc_fn, val: LabelledCollection, val_posteriors: np.ndarray):
        return cont_table.O_LEAP(acc_fn, KDEyML(MLPClassifier())).fit(val, val_posteriors)


class RQBScap(TMS_CAP):
    """
    Reverse Quantification-Based Sampling
    implemented using the CAP method
    """

    def __init__(
        self,
        clsf: ClsVariant,
        D: DatasetBundle,
        n_vsamples: int = 100,
        sample_size: int = None,
        aggr: Literal["mean", "median"] = "median",
    ):
        super().__init__(clsf, D)
        self.rqbs_params = dict(
            n_vsamples=n_vsamples,
            sample_size=sample_size,
            aggr=aggr,
        )

    @override
    def get_cap_model(self, acc_fn, val: LabelledCollection, val_posteriors: np.ndarray):
        return direct.RQBS(acc_fn, KDEyML(MLPClassifier()), **self.rqbs_params).fit(val, val_posteriors)


class PrediQuant(TMS_CAP):
    def __init__(
        self,
        clsf: ClsVariant,
        D: DatasetBundle,
        alpha=0.3,
        alpha_rate=1.2,
        sample_size: int = None,
        error: str | Callable = cap.error.mae,
        predict_train_prev=True,
    ):
        super().__init__(clsf, D)
        self.prediq_params = dict(
            alpha=alpha,
            alpha_rate=alpha_rate,
            sample_size=sample_size,
            error=error,
            predict_train_prev=predict_train_prev,
        )

    @override
    def get_cap_model(self, acc_fn: Callable, val: LabelledCollection, val_posteriors: np.ndarray):
        return direct.PrediQuant(
            acc=acc_fn,
            q=KDEyML(self.clsf.h),
            protocol=self.D.V2_prot,
            prot_posteriors=self.D.V2_prot_posteriors,
            **self.prediq_params,
        ).fit(val, val_posteriors)


class DoC(TMS_CAP):
    @override
    def get_cap_model(self, acc_fn: Callable, val: LabelledCollection, val_posteriors: np.ndarray):
        return direct.DoC(acc_fn, self.D.V2_prot, self.D.V2_prot_posteriors).fit(val, val_posteriors)


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
        sample_size=None,
    ):
        super().__init__(clsf, D)
        self.n_vsamples = n_vsamples
        self.sample_size = qp.environ["SAMPLE_SIZE"] if sample_size is None else sample_size

    @override
    def rank(self, acc_fn, val: LabelledCollection, val_posteriors: np.ndarray):
        if self.clsf.ms_ignore:
            return self.empty_rank()

        tinit = time()

        sidx = self.val_sidx_dict.get((self.D.dataset_name, self.D.n_classes), None)
        if sidx is None:
            q = KDEyML(MLPClassifier()).fit(val)
            q_hats = [q.quantify(Ui.X) for Ui in self.D.test_prot()]
            # normalize q_hats
            q_hats = [q_hat / q_hat.sum() for q_hat in q_hats]
            sidx = np.asarray(
                [[val.sampling_index(self.sample_size, *q_hat) for _ in range(self.n_vsamples)] for q_hat in q_hats]
            )
            self.val_sidx_dict[(self.D.dataset_name, self.D.n_classes)] = sidx

        ranking_vals = []
        for vidxs in sidx:
            accs = []
            for idx in vidxs:
                vali_yhat = val_posteriors[idx, :].argmax(axis=1)
                vali_y = val.y[idx]
                vaili_ct = contingency_table(vali_y, vali_yhat, self.D.n_classes)
                accs.append(acc_fn(vaili_ct))
            ranking_vals.append(np.median(accs))
        ranking_vals = np.array(ranking_vals)

        t_ave = (time() - tinit) / self.D.test_prot.total()

        return dict(
            ranking_vals=ranking_vals,
            t_ave=t_ave,
        )
