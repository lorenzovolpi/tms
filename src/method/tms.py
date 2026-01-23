from abc import abstractmethod
from time import time
from typing import Callable, Literal, override

import cap
import cap.models.cont_table as cont_table
import cap.models.direct as direct
import numpy as np
from cap.models.base import CAP
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from quapy.protocol import AbstractStochasticSeededProtocol
from sklearn.neural_network import MLPClassifier

from data import PreTrainedClassifier
from method.base import ModelSelectionMethod, NeedsValidationProtocol


class TMS(ModelSelectionMethod): ...


class TMS_CAP(TMS):
    @override
    def rank(self, h: PreTrainedClassifier, val: LabelledCollection, test_protocol: AbstractStochasticSeededProtocol):
        tinit = time()

        n_test_samples = test_protocol.total()
        val_posteriors = h.predict_proba(val.X)
        model = self.get_cap_model(self.acc, val, val_posteriors)
        test_prot_posteriors = [h.predict_proba(Ui.X) for Ui in test_protocol()]
        ranking_vals = self.get_ranking_vals(model, test_protocol, test_prot_posteriors)

        t_ave = (time() - tinit) / n_test_samples

        return dict(
            ranking_vals=ranking_vals,
            t_ave=t_ave,
        )

    @abstractmethod
    def get_cap_model(self, h: PreTrainedClassifier, val: LabelledCollection, val_posteriors: np.ndarray): ...

    def get_ranking_vals(
        self, model: CAP, test_protocol: AbstractStochasticSeededProtocol, test_prot_posteriors: np.ndarray
    ):
        return model.batch_predict(test_protocol, test_prot_posteriors)


class LEAP(TMS_CAP):
    @override
    def get_cap_model(self, h: PreTrainedClassifier, val: LabelledCollection, val_posteriors: np.ndarray):
        return cont_table.O_LEAP(self.acc, KDEyML(MLPClassifier())).fit(val, val_posteriors)


class RQBS(TMS_CAP):
    """
    Reverse Quantification-Based Sampling
    implemented using the CAP method
    """

    def __init__(
        self,
        acc: Callable,
        n_vsamples: int = 100,
        sample_size: int = None,
        aggr: Literal["mean", "median"] = "median",
    ):
        super().__init__(acc)
        self.rqbs_params = dict(
            n_vsamples=n_vsamples,
            sample_size=sample_size,
            aggr=aggr,
        )

    @override
    def get_cap_model(self, h: PreTrainedClassifier, val: LabelledCollection, val_posteriors: np.ndarray):
        return direct.RQBS(self.acc, KDEyML(MLPClassifier()), **self.rqbs_params).fit(val, val_posteriors)


class PrediQuant(TMS_CAP, NeedsValidationProtocol):
    def __init__(
        self,
        acc: Callable,
        alpha=0.3,
        alpha_rate=1.2,
        sample_size: int = None,
        error: str | Callable = cap.error.mae,
        predict_train_prev=True,
    ):
        super().__init__(acc)
        self.prediq_params = dict(
            alpha=alpha,
            alpha_rate=alpha_rate,
            sample_size=sample_size,
            error=error,
            predict_train_prev=predict_train_prev,
        )

    @override
    def get_cap_model(self, h: PreTrainedClassifier, val: LabelledCollection, val_posteriors: np.ndarray):
        return direct.PrediQuant(
            acc=self.acc,
            quantifier=KDEyML(self.clsf.h),
            protocol=self.val_protocol,
            prot_posteriors=self.get_val_prot_posteriors(h),
            **self.prediq_params,
        ).fit(val, val_posteriors)


class DoC(TMS_CAP, NeedsValidationProtocol):
    @override
    def get_cap_model(self, h: PreTrainedClassifier, val: LabelledCollection, val_posteriors: np.ndarray):
        return direct.DoC(
            acc_fn=self.acc,
            protocol=self.val_protocol,
            prot_posteriors=self.get_val_prot_posteriors(h),
        ).fit(val, val_posteriors)
