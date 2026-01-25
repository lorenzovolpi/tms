from time import time
from typing import Callable, Self, override

import numpy as np
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.protocol import AbstractStochasticSeededProtocol

from data import PreTrainedClassifier
from method.base import ModelSelectionMethod


class IMS(ModelSelectionMethod):
    def fit(
        self,
        val: LabelledCollection,
        val_posteriors: np.ndarray,
        test_protocol: AbstractStochasticSeededProtocol,
        test_prot_posteriors: list[np.ndarray],
    ) -> Self:
        n_classes = val.n_classes
        self.n_test_samples = test_protocol.total()
        y = val.y
        y_hat = np.argmax(val_posteriors, axis=1)
        ct = contingency_table(y, y_hat, n_classes)
        self._ct = ct
        return self

    @override
    def rank(self, acc: Callable[[np.ndarray], float]) -> list[float]:
        val_acc = acc(self._ct)
        ranking_vals = np.full(self.n_test_samples, val_acc).tolist()

        return ranking_vals
