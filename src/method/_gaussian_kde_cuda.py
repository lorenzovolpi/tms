import math
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm


class KernelDensityCuda:
    def __init__(
        self,
        bandwidth: float = 0.1,
        train_chunk_size: int = 5000,
        test_batch_size: int = 1000,
        return_type: Literal["np", "pt"] = "np",
    ):
        self.bandwidth = bandwidth
        self.train_chunk_size = train_chunk_size
        self.test_batch_size = test_batch_size
        self.train_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_type = return_type

    def fit(self, X: np.ndarray | torch.Tensor):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.train_data = X.to(self.device)
        return self

    def _chunked_log_densities(self, test_batch: torch.Tensor, n_train, n_features) -> torch.Tensor:
        batch_size = test_batch.shape[0]  # B x D
        batch_kernel = torch.zeros(batch_size, device=self.device)

        for j in range(0, n_train, self.train_chunk_size):
            train_chunk = self.train_data[j : j + self.train_chunk_size]  # T x D

            # INFO: compute the euclidean distance between test batch and train chunk
            # NOTE: this is the same as computing the broadcasting by hand as follows,
            # but much more efficient.
            # B x 1 x D - 1 x T x D = B x T x D
            # broadcasting: each sample in the test batch is subtracted with all
            # samples in the train chunk
            #
            # diffs = test_batch.unsqueeze(1) - train_chunk.unsqueeze(0)
            # sq_dists = (diffs**2).sum(dim=2)  # B x T
            sq_dists = torch.cdist(test_batch, train_chunk, p=2) ** 2

            # INFO: We take the exp of the squared distances in order to sum them
            # as we are computing kernel values in chunks
            kernel_values = torch.exp(-0.5 * sq_dists / (self.bandwidth**2))  # B x T

            # INFO: summed kernel values across all train chunk samples are added the the
            # kernels for the batch
            batch_kernel += kernel_values.sum(dim=1)  # B

        # INFO: log normalization factor for gaussian kernel
        # NOTE: it is equivalent to apply log to he following, but helps a little with
        # numerical stability
        #
        # norm_factor = (2 * np.pi * self.bandwidth**2) ** (n_features / 2)
        log_norm_factor = math.log(2 * np.pi * self.bandwidth**2) * math.log(n_features / 2)

        # INFO: average over all training samples and apply the norm_factor in log
        # NOTE: it is equivalent to applying log to the following, but helps a little with
        # numerical stability
        #
        # torch.log(batch_kernel / (n_train * norm_factor))
        log_density = torch.log(batch_kernel + 1e-10) - math.log(n_train) - log_norm_factor

        return log_density

    def _log_densities(self, test_batch: torch.Tensor, n_train, n_features) -> torch.Tensor:
        # B x T
        sq_dists = torch.cdist(test_batch, self.train_data, p=2) ** 2
        log_kernels = -0.5 * sq_dists / (self.bandwidth**2)
        # B
        batch_log_kernel = torch.logsumexp(log_kernels, dim=1)

        log_norm_factor = math.log(2 * np.pi * self.bandwidth**2) * math.log(n_features / 2)
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

            if self.train_chunk_size is None:
                log_density = self._log_densities(test_batch, n_train, n_features)
            else:
                log_density = self._chunked_log_densities(test_batch, n_train, n_features)
            log_densities.append(log_density)

        log_densities = torch.cat(log_densities)

        if self.return_type == "np":
            return log_densities.cpu().numpy()

        return log_densities
