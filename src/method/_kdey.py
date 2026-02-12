import cvxpy as cp
import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from sklearn.base import BaseEstimator


def _optim_cvxpy(n_classes: int, test_densities: np.ndarray, epsilon: float) -> np.ndarray:
    x = cp.Variable(n_classes)
    test_mixture_likelihood = test_densities @ x

    objective = cp.Minimize(-cp.sum(cp.log(test_mixture_likelihood + epsilon)))
    constraints = [cp.sum(x) == 1, x >= 0, x <= 1, test_mixture_likelihood >= 1e-8]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()
    except cp.SolverError:
        prob.solve(solver=cp.SCS)

    return x.value


class KDEyMLFast(KDEyML):
    def __init__(
        self,
        classifier: BaseEstimator = None,
        val_split=5,
        bandwidth=0.1,
        limit_kde_train: bool | int | float = False,
        random_state=None,
    ):
        super().__init__(classifier, val_split, bandwidth, random_state)
        self.limit_kde_train = self._get_kde_limit(limit_kde_train)

    def _get_kde_limit(self, limit):
        if (isinstance(limit, int) or isinstance(limit, float)) and limit > 0:
            return int(limit) if limit > 1 else limit
        elif isinstance(limit, bool) and limit:
            return int(1e4)

        return None

    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        _limit = self.limit_kde_train or 0
        if _limit <= 1:
            _limit = int(len(classif_predictions) * _limit)

        if _limit > 0 and _limit < len(classif_predictions):
            classif_predictions = classif_predictions.uniform_sampling(size=_limit, random_state=self.random_state)

        self.mix_densities = self.get_mixture_components(*classif_predictions.Xy, data.classes_, self.bandwidth)
        return self

    def aggregate(self, posteriors: np.ndarray):
        """
        Searches for the mixture model parameter (the sought prevalence values) that maximizes the likelihood
        of the data (i.e., that minimizes the negative log-likelihood)

        :param posteriors: instances in the sample converted into posterior probabilities
        :return: a vector of class prevalence estimates
        """
        with qp.util.temp_seed(self.random_state):
            epsilon = 1e-6
            n_classes = len(self.mix_densities)
            test_densities = [self.pdf(kde_i, posteriors) for kde_i in self.mix_densities]

            return _optim_cvxpy(n_classes, test_densities, epsilon)
