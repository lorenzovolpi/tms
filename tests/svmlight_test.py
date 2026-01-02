from typing import Tuple

import cap
import numpy as np
import pandas as pd
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset
from cap.error import vanilla_acc
from cap.models import O_LEAP, DoC
from cap.models.base import ClassifierAccuracyPrediction
from cap.utils.commons import true_acc
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import KDEyML
from sklearn.neural_network import MLPClassifier

from svmlight import SVMlight
from util import get_test_prot, split_validation

qp.environ["SAMPLE_SIZE"] = 100
qp.environ["_R_SEED"] = 0
N_SAMPLES = 1000


def test_svmlight():
    acc_name, acc_fn = "vanilla_accuracy", vanilla_acc
    h = SVMlight(kernel="rbf")
    results = []
    for d in UCI_BINARY_DATASETS[:1]:
        L, V, U = fetch_UCIBinaryDataset(d)
        h.fit(*L.Xy)

        print(d)
        print(h.predict(U.X))

    #     test_prot = get_test_prot(U, repeats=N_SAMPLES)
    #     V1, V2_prot = split_validation(V)
    #
    #     test_prot_post = [h.predict(Ui.X) for Ui in test_prot()]
    #     V2_prot_post = [h.predict(V2i.X) for V2i in V2_prot()]
    #     V_post = h.predict(V.X)
    #     V1_post = h.predict(V1.X)
    #
    #     methods: dict[str, Tuple[ClassifierAccuracyPrediction, LabelledCollection, np.ndarray]] = {
    #         "DoC": (DoC(vanilla_acc, V2_prot, V2_prot_post), V1, V1_post),
    #         "O-LEAP": (O_LEAP(acc_fn, KDEyML(MLPClassifier())), V, V_post),
    #     }
    #     true_accs = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]
    #
    #     for method_name, (method, val, val_post) in methods.items():
    #         method.fit(val, val_post)
    #         estim_accs = [method.predict(Ui.X, P) for Ui, P in zip(test_prot(), test_prot_post)]
    #         acc_errs = cap.error.ae(true_accs, estim_accs)
    #         results.append(
    #             dict(
    #                 method=[method_name] * N_SAMPLES,
    #                 dataset=[d] * N_SAMPLES,
    #                 classifier=["svmlight"] * N_SAMPLES,
    #                 acc=[acc_name] * N_SAMPLES,
    #                 true_accs=true_accs,
    #                 estim_accs=estim_accs,
    #                 acc_errs=acc_errs,
    #             )
    #         )
    #
    # df = pd.concat([pd.DataFrame.from_dict(d) for d in results])
    #
    # print(pd.pivot_table(df, columns="method", index="dataset", values="acc_errs"))


if __name__ == "__main__":
    test_svmlight()
