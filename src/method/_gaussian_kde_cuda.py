import math
from typing import Literal

import numpy as np
import torch


class KernelDensityCuda:
    def __init__(
        self,
        bandwidth: float = 0.1,
        test_batch_size: int = 1000,
        return_type: Literal["np", "pt"] = "np",
    ):
        self.bandwidth = bandwidth
        self.test_batch_size = test_batch_size
        self.train_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_type = return_type

    def fit(self, X: np.ndarray | torch.Tensor):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.train_data = X.to(self.device)
        return self

    def _log_densities(self, test_batch: torch.Tensor, n_train, n_features) -> torch.Tensor:
        #
        # INFO: compute the euclidean distance between test batch and train; this is the
        # same as computing the broadcasting by hand as follows, but much more efficient:
        # B x 1 x D - 1 x T x D = B x T x D
        # broadcasting: each sample in the test batch is subtracted with all
        # samples in the train chunk
        #
        # diffs = test_batch.unsqueeze(1) - self.train_data.unsqueeze(0)
        # sq_dists = (diffs**2).sum(dim=2)  # B x T
        #
        sq_dists = torch.cdist(test_batch, self.train_data, p=2) ** 2  # B x T

        # INFO: We take the negative squared distances divided by 2 and the squared bandwidth
        # and see these values as log kernel values. Then we perform logsumexp is computed
        # along the dimension of size T (the dimension of the train's length). This is
        # equivalent to the following code, but more efficient in terms of numerical stability.
        #
        # kernel_values = torch.exp(-0.5 * sq_dists / (self.bandwidth**2))  # B x T
        # batch_log_kernel = torch.log(kernel_values.sum(dim=1))  # B
        #
        log_kernels = -0.5 * sq_dists / (self.bandwidth**2)
        batch_log_kernel = torch.logsumexp(log_kernels, dim=1)

        # INFO: log normalization factor for gaussian kernel; this is equivalent to
        # apply log to the following, but helps a little with numerical stability.
        #
        # norm_factor = (2 * np.pi * self.bandwidth**2) ** (n_features / 2)
        #
        log_norm_factor = math.log(2 * np.pi * self.bandwidth**2) * math.log(n_features / 2)

        # INFO: average over all training samples and apply the norm_factor in log; it is
        # equivalent to applying log to the following, but helps with numerical stability.
        #
        # torch.log(batch_kernel / (n_train * norm_factor))
        #
        log_density = batch_log_kernel - math.log(n_train) - log_norm_factor

        return log_density

    def score_samples(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)

        n_test = X.shape[0]
        n_train = self.train_data.shape[0]
        n_features = X.shape[1]

        log_densities = []

        for i in range(0, n_test, self.test_batch_size):
            test_batch = X[i : i + self.test_batch_size]

            log_density = self._log_densities(test_batch, n_train, n_features)
            log_densities.append(log_density)

        log_densities = torch.cat(log_densities)

        if self.return_type == "np":
            return log_densities.cpu().numpy()

        return log_densities
