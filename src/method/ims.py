from time import time
from typing import override

import numpy as np
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection

from method.base import ModelSelection


class IMS(ModelSelection):
    @override
    def rank(self, acc_fn, val: LabelledCollection, val_posteriors: np.ndarray):
        if self.clsf.ms_ignore:
            return self.empty_rank()

        tinit = time()

        y = val.y
        y_hat = np.argmax(val_posteriors, axis=1)
        ct = contingency_table(y, y_hat, self.D.n_classes)
        val_acc = acc_fn(ct)
        ranking_vals = np.full(self.D.test_prot.total(), val_acc).tolist()

        t_ave = (time() - tinit) / self.D.test_prot.total()

        return dict(
            ranking_vals=ranking_vals,
            t_ave=t_ave,
        )
