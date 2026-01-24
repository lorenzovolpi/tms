from time import time

import numpy as np
import quapy as qp
from cap.data.datasets import fetch_UCIMulticlassDataset
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from data import PreTrainedClassifier
from util import split_validation

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0
NUM_TEST = 1000

if __name__ == "__main__":
    # h = LogisticRegression()
    h = SVC(kernel="rbf", probability=True)
    # h = MLPClassifier()
    # h = KNN(n_neighbors=13)
    L, V, U = fetch_UCIMulticlassDataset("mhr")
    h.fit(*L.Xy)

    t0 = time()
    V_posteriors = h.predict_proba(V.X)
    U_posteriors = h.predict_proba(U.X)
    t_UV = time() - t0
    print("Time for UV posteriors:", t_UV)

    h_params = h.get_params()

    ph = PreTrainedClassifier(U.X, U_posteriors, V.X, V_posteriors, "LR", h_params, True, False)

    V1, V2_prot = split_validation(V, random_state=qp.environ["_R_SEED"])
    test_prot = qp.protocol.UPP(
        U,
        repeats=NUM_TEST,
        return_type="labelled_collection",
        random_state=qp.environ["_R_SEED"],
    )

    t0 = time()
    h_test_prot_posteriors = [h.predict_proba(Ui.X) for Ui in test_prot()]
    h_V1_posteriors = h.predict_proba(V1.X)
    h_V2_prot_posteriors = [h.predict_proba(Ui.X) for Ui in V2_prot()]
    t_h = time() - t0
    print(f"Time for h predictions: {t_h:.4f}s")

    t0 = time()
    ph_test_prot_posteriors = [ph.predict_proba(Ui.X) for Ui in test_prot()]
    ph_V1_posteriors = ph.predict_proba(V1.X)
    ph_V2_prot_posteriors = [ph.predict_proba(Ui.X) for Ui in V2_prot()]
    t_ph = time() - t0
    print(f"Time for ph predictions: {t_ph:.4f}s")

    for p1, p2 in zip(h_V1_posteriors, ph_V1_posteriors):
        if not np.isclose(p1, p2).all():
            print(p1, p2)
    assert all([np.isclose(p1, p2).all() for p1, p2 in zip(h_test_prot_posteriors, ph_test_prot_posteriors)])
    assert all([np.isclose(p1, p2).all() for p1, p2 in zip(h_V2_prot_posteriors, ph_V2_prot_posteriors)])
    assert np.isclose(h_V1_posteriors, ph_V1_posteriors).all()
    print("ALL CHECKS PASSED.")
