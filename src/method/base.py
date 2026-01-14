from abc import ABC, abstractmethod

import numpy as np
from quapy.data import LabelledCollection

from data import ClsVariant, DatasetBundle


class ModelSelection(ABC):
    def __init__(self, clsf: ClsVariant, D: DatasetBundle):
        self.clsf = clsf
        self.D = D

    @abstractmethod
    def rank(self, acc_fn, val: LabelledCollection, val_posteriors: np.ndarray): ...

    def empty_rank(self):
        return dict(
            ranking_vals=[None] * self.D.test_prot.total(),
            t_train=None,
            t_test_ave=None,
        )
