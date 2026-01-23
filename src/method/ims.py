from time import time
from typing import override

import numpy as np
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.protocol import AbstractStochasticSeededProtocol

from data import PreTrainedClassifier
from method.base import ModelSelectionMethod


class IMS(ModelSelectionMethod):
    @override
    def rank(self, h: PreTrainedClassifier, val: LabelledCollection, test_protocol: AbstractStochasticSeededProtocol):
        n_test_samples = test_protocol.total()
        val_posteriors = h.predict_proba(val.X)
        tinit = time()

        y = val.y
        y_hat = np.argmax(val_posteriors, axis=1)
        ct = contingency_table(y, y_hat, self.D.n_classes)
        val_acc = self.acc(ct)
        ranking_vals = np.full(n_test_samples, val_acc).tolist()

        t_ave = (time() - tinit) / n_test_samples

        return dict(
            ranking_vals=ranking_vals,
            t_ave=t_ave,
        )
