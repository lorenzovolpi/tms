from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from quapy.data import LabelledCollection
from quapy.protocol import AbstractStochasticSeededProtocol

from data import PreTrainedClassifier


class ModelSelectionMethod(ABC):
    def __init__(self, acc) -> None:
        self.acc = acc

    @abstractmethod
    def fit(
        self,
        val: LabelledCollection,
        val_posteriors: np.ndarray,
        test_protocol: AbstractStochasticSeededProtocol,
        test_prot_posteriors: list[np.ndarray],
    ): ...

    @abstractmethod
    def rank(self, acc: Callable[[np.ndarray], float]) -> list[float]: ...

    def empty_rank(self):
        return dict(
            ranking_vals=[None] * self.D.test_prot.total(),
            t_ave=None,
        )


class NeedsValidationProtocol:
    """
    Interface that indicates that the model selection method requires a validation protocol
    """

    def set_validation_protocol(
        self, val_protocol: AbstractStochasticSeededProtocol, val_prot_posteriors: list[np.ndarray]
    ):
        self.val_protocol = val_protocol
        self.val_prot_posteriors = val_prot_posteriors
