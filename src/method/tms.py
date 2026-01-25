from typing import Callable, Literal, Self, override

import cap
import cap.models.cont_table as cont_table
import cap.models.direct as direct
import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from quapy.protocol import AbstractStochasticSeededProtocol
from sklearn.neural_network import MLPClassifier

from method.base import ModelSelectionMethod, NeedsValidationProtocol


class TMS(ModelSelectionMethod): ...


class LEAP(TMS):
    @override
    def fit(
        self,
        val: LabelledCollection,
        val_posteriors: np.ndarray,
        test_protocol: AbstractStochasticSeededProtocol,
        test_prot_posteriors: list[np.ndarray],
    ) -> Self:
        self.model = cont_table.O_LEAP(self.acc, KDEyML(MLPClassifier())).fit(val, val_posteriors)
        self._cts = self.model._batch_predict_ct(test_protocol, test_prot_posteriors)
        return self

    @override
    def rank(self, acc: Callable[[np.ndarray], float]) -> list[float]:
        self.model.acc_fn = acc
        ranking_vals = [acc(ct) for ct in self._cts]
        return ranking_vals


class RQBS(TMS):
    """
    Reverse Quantification-Based Sampling
    implemented using the CAP method
    """

    def __init__(
        self,
        acc: Callable[[np.ndarray], float],
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
    def fit(
        self,
        val: LabelledCollection,
        val_posteriors: np.ndarray,
        test_protocol: AbstractStochasticSeededProtocol,
        test_prot_posteriors: list[np.ndarray],
    ) -> Self:
        self.model = direct.RQBS(self.acc, KDEyML(MLPClassifier()), **self.rqbs_params).fit(val, val_posteriors)
        self.tp_val_sample_cts = [self.model._predict_val_sample_cts(Ui.X) for Ui in test_protocol()]
        return self

    @override
    def rank(self, acc: Callable[[np.ndarray], float]) -> list[float]:
        self.model.acc = acc
        ranking_vals = []
        for vs_cts in self.tp_val_sample_cts:
            vs_accs = self.model._predict_from_val_cts(vs_cts)
            ranking_vals.append(self.model.aggr_fun(vs_accs))

        return ranking_vals


class PrediQuant(TMS, NeedsValidationProtocol):
    def __init__(
        self,
        acc: Callable[[np.ndarray], float],
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
    def fit(
        self,
        val: LabelledCollection,
        val_posteriors: np.ndarray,
        test_protocol: AbstractStochasticSeededProtocol,
        test_prot_posteriors: list[np.ndarray],
    ) -> Self:
        self.model = direct.PrediQuant(
            acc=self.acc,
            quantifier=KDEyML(MLPClassifier()),
            protocol=self.val_protocol,
            prot_posteriors=self.val_prot_posteriors,
            **self.prediq_params,
        ).fit(val, val_posteriors)
        self.tp_test_priors = [self.model._predict_test_priors(Ui.X) for Ui in test_protocol()]
        return self

    @override
    def rank(self, acc: Callable[[np.ndarray], float]) -> list[float]:
        self.model.acc = acc
        ranking_vals = []
        for priors in self.tp_test_priors:
            ranking_vals.append(self.model._predict_from_test_priors(priors))

        return ranking_vals


class DoC(TMS, NeedsValidationProtocol):
    @override
    def fit(
        self,
        val: LabelledCollection,
        val_posteriors: np.ndarray,
        test_protocol: AbstractStochasticSeededProtocol,
        test_prot_posteriors: list[np.ndarray],
    ) -> Self:
        self.val = val
        self.val_posteriors = val_posteriors
        self.test_protocol = test_protocol
        self.test_prot_posteriors = test_prot_posteriors

    @override
    def rank(self, acc: Callable[[np.ndarray], float]) -> list[float]:
        model = direct.DoC(
            acc_fn=acc,
            protocol=self.val_protocol,
            prot_posteriors=self.val_prot_posteriors,
        ).fit(self.val, self.val_posteriors)
        ranking_vals = model.batch_predict(self.test_protocol, self.test_prot_posteriors)
        return ranking_vals
