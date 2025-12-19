import numpy as np
import pytest
import quapy as qp
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import k_bin, k_macro
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import contingency_matrix

qp.environ["_R_SEED"] = 0


@pytest.fixture
def clsf():
    yield "LR", LogisticRegression()


def data_bin_params():
    return ["spambase"]


@pytest.fixture(params=data_bin_params())
def data_bin(request):
    name = request.param
    L, _, U = fetch_UCIBinaryDataset(name)
    test_prot = UPP(U, sample_size=1000, repeats=100, random_state=0, return_type="labelled_collection")
    yield (name, L, test_prot)


def data_multi_params():
    return ["molecular", "isolet"]


@pytest.fixture(params=data_multi_params())
def data_multi(request):
    name = request.param
    L, _, U = fetch_UCIMulticlassDataset(name)
    test_prot = UPP(U, sample_size=1000, repeats=100, random_state=0, return_type="labelled_collection")
    yield (name, L, test_prot)


class TestKMeasure:
    def test_k_bin(self, clsf, data_bin):
        h_name, h = clsf
        data_name, L, test_prot = data_bin
        h.fit(*L.Xy)

        y = [Ui.y for Ui in test_prot()]
        P = [h.predict_proba(Ui.X) for Ui in test_prot()]
        y_hat = [np.argmax(ph, axis=1) for ph in P]
        ct = [contingency_matrix(yi, yhi) / len(y) for yi, yhi in zip(y, y_hat)]
        k = [k_bin(cti) for cti in ct]
        assert all(ki >= -1 and ki <= 1 for ki in k)
        k_mean = np.mean(k)
        assert -1 <= k_mean <= 1
        print(f"{h_name} on {data_name}: k_bin mean = {k_mean:.4f}")

    def test_k_macro(self, clsf, data_multi):
        h_name, h = clsf
        data_name, L, test_prot = data_multi
        h.fit(*L.Xy)

        y = [Ui.y for Ui in test_prot()]
        P = [h.predict_proba(Ui.X) for Ui in test_prot()]
        y_hat = [np.argmax(ph, axis=1) for ph in P]
        ct = [contingency_matrix(yi, yhi) / len(y) for yi, yhi in zip(y, y_hat)]
        k = [k_macro(cti) for cti in ct]
        assert all(ki >= -1 and ki <= 1 for ki in k)
        k_mean = np.mean(k)
        assert -1 <= k_mean <= 1
        print(f"{h_name} on {data_name}: k_bin mean = {k_mean:.4f}")
