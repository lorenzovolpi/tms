from abc import ABC, abstractmethod
from typing import Callable

from quapy.data import LabelledCollection
from quapy.protocol import AbstractStochasticSeededProtocol

from data import PreTrainedClassifier


class ModelSelectionMethod(ABC):
    def __init__(self, acc: Callable):
        self.acc = acc

    @abstractmethod
    def rank(
        self, h: PreTrainedClassifier, val: LabelledCollection, test_protocol: AbstractStochasticSeededProtocol
    ): ...

    def empty_rank(self):
        return dict(
            ranking_vals=[None] * self.D.test_prot.total(),
            t_ave=None,
        )


class NeedsValidationProtocol:
    """
    Interface that indicates that the model selection method requires a validation protocol
    """

    def set_validation_protocol(self, val_protocol: AbstractStochasticSeededProtocol):
        self.val_protocol = val_protocol

    def get_val_prot_posteriors(self, h: PreTrainedClassifier):
        return [h.predict_proba(Ui.X) for Ui in self.val_protocol()]
