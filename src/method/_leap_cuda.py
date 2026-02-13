from typing import Callable, Literal, Self, TypeAlias

import numpy as np
import quapy.functional as F
import torch
from cap.utils.commons import contingency_table

from method._kdeyml_cuda import KDEyMLCuda

Vector: TypeAlias = np.ndarray | torch.Tensor


class OLEAPCuda:
    def __init__(
        self,
        kdey_params: dict = None,
    ):
        self.kdey_params = kdey_params or dict(
            bandwidth=0.1,
            return_type="pt",
        )

    def _build_system(self, y: np.ndarray, X_post: np.ndarray):
        n_unknowns = self.n**2
        n_eqs = (self.n + 1) ** 2

        idx = np.arange(self.n**2).reshape(self.n, self.n)

        y_hat = np.argmax(X_post, axis=1)
        ct = contingency_table(y, y_hat, self.n)
        class_cond_ratios = ct / ct.sum(axis=1, keepdims=True)

        A_1 = np.ones(n_unknowns).reshape(1, -1)

        A_2 = np.zeros(self.n**2, n_unknowns)
        for k in range(self.n**2):
            i, j = k // self.n, k % self.n
            ratio_ij = class_cond_ratios[i, j]
            A_2[k, idx[i, :]] = -ratio_ij
            A_2[k, idx[i, j]] = 1 - ratio_ij

        A_3 = np.zeros(self.n, n_unknowns)
        for i in range(self.n):
            A_3[i, idx[:, i]] = 1

        A_4 = np.zeros(self.n, n_unknowns)
        for i in range(self.n):
            A_4[i, idx[i, :]] = 1

        A = np.vstack([A_1, A_2, A_3, A_4])
        self.A = torch.from_numpy(A).float().to(self.device)

        b = np.zeros(n_eqs)
        b[0] = 1
        self.b = torch.from_numpy(b).float()

    def fit(self, X: np.ndarray, y: np.ndarray, X_post: np.ndarray) -> Self:
        self.classes_ = np.unique(y)
        self.n = self.classes_.shape[0]

        self._build_system(y, X_post)
        self.q = KDEyMLCuda(**self.kdey_params).fit(X, y)

        return self

    def predict_ct(self, X: np.ndarray, X_post: np.ndarray) -> torch.Tensor:
        h_preds = np.argmax(X_post, axis=1)

        cc_prev_estim = torch.from_numpy(F.prevalence_from_labels(h_preds, self.classes_)).float().to(self.device)
        q_prev_estim = self.q.quantify(X)

        b = self.b.to(self.device)
        b[-2 * self.n : -self.n] = cc_prev_estim
        b[-self.n :] = q_prev_estim

        ct = self._optimize(b).view(self.n, self.n)

        return ct

    def batch_predict_ct(self, Xs: np.ndarray, Xs_post: np.ndarray) -> torch.Tensor:
        bs = []
        for i in range(Xs.shape[0]):
            X, X_post = Xs[i], Xs_post[i]

            h_preds = np.argmax(X_post, axis=1)

            cc_prev_estim = torch.from_numpy(F.prevalence_from_labels(h_preds, self.classes_)).float().to(self.device)
            q_prev_estim = self.q.quantify(X)

            b = self.b.clone().to(self.device)
            b[-2 * self.n : -self.n] = cc_prev_estim
            b[-self.n :] = q_prev_estim

            bs.append(b)

        bs = torch.stack(bs, dim=0)

        cts = self._optimize_batched(bs).view(-1, self.n, self.n)

        return cts

    def with_acc(self, acc: Callable) -> Self:
        self._acc = acc
        return self

    def predict(self, X: np.ndarray, X_Post: np.ndarray) -> float:
        if not hasattr(self, "_acc"):
            raise ValueError("Accuracy function is not set.")

        ct = self.predict_ct(X, X_Post).cpu().numpy()
        return self._acc(ct)

    def batch_predict(self, Xs: np.ndarray, Xs_post: np.ndarray) -> list[float]:
        if not hasattr(self, "_acc"):
            raise ValueError("Accuracy function is not set.")

        cts = self.batch_predict_ct(Xs, Xs_post)
        cts = [cts[i].cpu().numpy() for i in range(cts.shape[0])]
        return list(map(self._acc, cts))

    def _optimize(self, b: torch.Tensor) -> torch.Tensor:
        n_unk = self.A.shape[1]
        x = torch.full((n_unk,), 1 / n_unk, dtype=torch.float32, device=self.device, requires_grad=True)
        lambda_lagrange = torch.nn.Parameter(torch.tensor(0.0, device=self.device))

        # hyperparameters
        lr_x = 5e-3
        lr_lambda = 1e-1
        num_iters = 5000

        x_opt = torch.optim.Adam([x], lr=lr_x)
        lambda_opt = torch.optim.SGD([lambda_lagrange], lr=lr_lambda)

        for _ in range(num_iters):
            x_opt.zero_grad()
            lambda_opt.zero_grad()
            # loss
            loss = torch.norm(self.A @ x - b) ** 2
            # penalty to enforce x.sum() == 1 with lagrangian
            loss += lambda_lagrange.data * (x.sum() - 1)
            # step
            loss.backward()
            x_opt.step()
            lambda_opt.step()
            # apply bounds
            with torch.no_grad():
                x.clamp_(0, 1)

        x = x.clone().detach()
        x /= x.sum()

        return x

    def _optimize_batched(self, bs: torch.Tensor) -> torch.Tensor:
        k, n_unk = bs.shape[0], self.A.shape[1]

        xs = torch.full((k, n_unk), 1 / n_unk, dtype=torch.float32, device=self.device, requires_grad=True)

        lambda_lagrange = torch.nn.Parameter(torch.zeros(k, device=self.device))

        # hyperparameters
        lr_xs = 5e-3
        lr_lambda = 1e-1
        num_iters = 5000

        xs_opt = torch.optim.Adam([xs], lr=lr_xs)
        lambda_opt = torch.optim.SGD([lambda_lagrange], lr=lr_lambda)

        for _ in range(num_iters):
            xs_opt.zero_grad()
            lambda_opt.zero_grad()
            # losses for all batches
            Ax = torch.matmul(xs, self.A.T)
            losses = torch.norm(Ax - bs, dim=1) ** 2
            # constraint
            constraints = xs.sum(dim=1) - 1
            # cumulative loss
            loss = losses.sum() + (lambda_lagrange.data * constraints).sum()
            # step
            loss.backward()
            xs_opt.step()
            lambda_opt.step()
            # apply bounds
            with torch.no_grad():
                xs.clamp_(0, 1)

        xs = xs.clone().detach()
        xs = xs / torch.sum(torch.abs(xs), dim=1, keepdim=True)

        return xs
