from multiprocessing import Pool
from typing import Callable, Literal, Self, TypeAlias

import cvxpy as cp
import numpy as np
import quapy.functional as F
import torch
from cap.utils.commons import contingency_table
from scipy import sparse
from tqdm import tqdm

from method._kdeyml_cuda import KDEyMLCuda

Vector: TypeAlias = np.ndarray | torch.Tensor


class OLEAPCuda:
    def __init__(
        self,
        kdey_params: dict = None,
        max_optim_iter: int = 300,
        solver: Literal["cuda", "cuda_sparse", "cvxpy", "cvxpy_sparse", "ratios"] = "ratios",
    ):
        self.kdey_params = kdey_params or dict(
            bandwidth=0.1,
            return_type="np",
        )
        self.max_optim_iter = max_optim_iter
        self.solver = solver

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_system(self, y: np.ndarray, X_post: np.ndarray):
        n_unknowns = self.n**2
        n_eqs = (self.n + 1) ** 2

        idx = np.arange(self.n**2).reshape(self.n, self.n)

        y_hat = np.argmax(X_post, axis=1)
        ct = contingency_table(y, y_hat, self.n)
        class_cond_ratios = ct / ct.sum(axis=1, keepdims=True)

        A_1 = np.ones(n_unknowns).reshape(1, -1)

        A_2 = np.zeros((self.n**2, n_unknowns))
        for k in range(self.n**2):
            i, j = k // self.n, k % self.n
            ratio_ij = class_cond_ratios[i, j]
            A_2[k, idx[i, :]] = -ratio_ij
            A_2[k, idx[i, j]] = 1 - ratio_ij

        A_3 = np.zeros((self.n, n_unknowns))
        for i in range(self.n):
            A_3[i, idx[:, i]] = 1

        A_4 = np.zeros((self.n, n_unknowns))
        for i in range(self.n):
            A_4[i, idx[i, :]] = 1

        self.A = np.vstack([A_1, A_2, A_3, A_4])

        self.b = np.zeros(n_eqs)
        self.b[0] = 1

    def fit(self, X: np.ndarray, y: np.ndarray, X_post: np.ndarray) -> Self:
        self.classes_ = np.unique(y)
        self.n = self.classes_.shape[0]

        self._build_system(y, X_post)
        self.q = KDEyMLCuda(**self.kdey_params).fit(X, y)
        self.train_prev = F.prevalence_from_labels(y, self.classes_)
        self.train_ct = contingency_table(y, np.argmax(X_post, axis=1), self.n)

        return self

    def predict_ct(self, X: np.ndarray, X_post: np.ndarray) -> np.ndarray:
        h_preds = np.argmax(X_post, axis=1)

        cc_prev_estim = torch.from_numpy(F.prevalence_from_labels(h_preds, self.classes_)).float().to(self.device)
        q_prev_estim = self.q.quantify(X)

        A = torch.from_numpy(self.A).float().to(self.device)
        b = torch.from_numpy(self.b).float().to(self.device)
        b[-2 * self.n : -self.n] = cc_prev_estim
        b[-self.n :] = q_prev_estim

        ct = self._optimize_cuda(A, b).view(self.n, self.n)

        return ct.cpu().numpy()

    def batch_predict_ct(self, Xs: list[np.ndarray], Xs_post: list[np.ndarray]) -> np.ndarray:
        bs = []
        q_prev_estims = self.q.batch_quantify(Xs)
        if self.solver == "ratios":
            cts_hat = []
            for i in range(len(Xs)):
                adjustment = q_prev_estims[i] / self.train_prev
                cts_hat.append(self.train_ct * adjustment[:, np.newaxis])

            return np.stack(cts_hat)

        for i in range(len(Xs)):
            X_post = Xs_post[i]

            h_preds = np.argmax(X_post, axis=1)

            cc_prev_estim = F.prevalence_from_labels(h_preds, self.classes_)

            b = self.b.copy()
            b[-2 * self.n : -self.n] = cc_prev_estim
            b[-self.n :] = q_prev_estims[i]

            bs.append(b)

        bs = np.stack(bs, axis=0)

        cts = self._optimize_batched(self.A, bs).reshape(-1, self.n, self.n)

        return cts

    def with_acc(self, acc: Callable) -> Self:
        self._acc = acc
        return self

    def predict(self, X: np.ndarray, X_Post: np.ndarray) -> float:
        if not hasattr(self, "_acc"):
            raise ValueError("Accuracy function is not set.")

        ct = self.predict_ct(X, X_Post)
        return self._acc(ct)

    def batch_predict(self, Xs: list[np.ndarray], Xs_post: list[np.ndarray]) -> list[float]:
        if not hasattr(self, "_acc"):
            raise ValueError("Accuracy function is not set.")

        cts = self.batch_predict_ct(Xs, Xs_post)
        cts = [cts[i] for i in range(cts.shape[0])]
        return list(map(self._acc, cts))

    def _optimize_cuda(self, A: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        n_unk = A.shape[1]
        logits = torch.zeros(n_unk, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([logits], lr=0.5, betas=(0.9, 0.999))

        best_loss = float("inf")
        best_x = None
        tol = 1e-6
        patience = 20
        no_improve = 0
        eval_iter = 1

        for _iter in range(self.max_optim_iter):
            optimizer.zero_grad()

            x = torch.softmax(logits, dim=0)
            Ax = torch.matmul(A, x)
            loss = torch.sum((Ax - b) ** 2)

            loss.backward()
            optimizer.step()

            if _iter % eval_iter == 0:
                current_loss = loss.item()
                if current_loss < best_loss - tol:
                    best_loss = current_loss
                    best_x = x.detach().clone()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience // eval_iter:
                    break

        x = best_x if best_x is not None else x.detach()
        return (x / x.sum()).cpu().numpy()

    def _optimize_cuda_sparse(self, A: torch.tensor, b: torch.Tensor) -> np.ndarray:
        n_unk = A.shape[1]
        reg_lambda = 1e-4

        z = torch.zeros(n_unk, device=self.device, requires_grad=True)

        optimizer = torch.optim.LBFGS([z], lr=1, max_iter=50, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()

            x = torch.softmax(z, dim=0)
            residual = torch.mv(self.A, x) - b
            data_loss = torch.sum(residual**2)

            l2_penalty = reg_lambda * torch.sum(x**2)

            total_loss = data_loss + l2_penalty
            total_loss.backward()
            return total_loss

        for i in range(5):
            optimizer.step(closure)

        x = torch.softmax(z.detach().clone(), dim=0)
        return x.cpu().numpy()

    def _optimize_cvxpy(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = cp.Variable(A.shape[1])

        objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b))
        constraints = [x >= 0, x <= 1, cp.sum(x) == 1]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        return x.value

    def _optimize_cvxpy_sparse(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        x = cp.Variable(A.shape[1])
        lambda_reg = 1e-4

        objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambda_reg * cp.sum_squares(x))
        constraints = [x >= 0, cp.sum(x) == 1]

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)

        return x.value

    def _optimize_batched(self, A: np.ndarray, bs: np.ndarray) -> np.ndarray:
        # if self.solver == "cvxpy_sparse":
        #     all_xs = _optimize_batched_cvxpy_sparse(A, bs)
        #     return np.stack(all_xs, axis=0)
        # if self.solver == "cvxpy_sparse":
        #     A = sparse.csr_matrix(self.A)
        #     all_xs = _optimize_compiled_cvxpy_sparse(A, bs)
        #     return np.stack(all_xs, axis=0)

        if self.solver == "cuda":
            A = torch.from_numpy(self.A).float().to(self.device)
            bs = torch.from_numpy(bs).float().to(self.device)
        elif self.solver == "cuda_sparse":
            A = torch.from_numpy(sparse.csr_matrix(self.A)).float().to(self.device)
            bs = torch.from_numpy(bs).float().to(self.device)
        elif self.solver == "cvxpy_sparse":
            A = sparse.csr_matrix(self.A)
        else:
            A = self.A

        all_xs = []
        for k in tqdm(range(bs.shape[0]), desc="leap optim"):
            if self.solver == "cuda":
                all_xs.append(self._optimize_cuda(A, bs[k]).cpu().numpy())
            elif self.solver == "cuda_sparse":
                all_xs.append(self._optimize_cuda_sparse(A, bs[k]))
            elif self.solver == "cvxpy":
                all_xs.append(self._optimize_cvxpy(A, bs[k]))
            elif self.solver == "cvxpy_sparse":
                all_xs.append(self._optimize_cvxpy_sparse(A, bs[k]))
        xs = np.stack(all_xs, axis=0)

        return xs


def _worker(b_val):
    n_unk = A_sparse.shape[1]
    x = cp.Variable(n_unk)

    obj = cp.Minimize(0.5 * cp.sum_squares(A_sparse @ x - b_val) + 1e-4 * cp.sum_squares(x))
    prob = cp.Problem(obj, [cp.sum(x) == 1, x >= 0])
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=10000)
    return x.value


def _optimize_batched_cvxpy_sparse(A: np.ndarray, bs: np.ndarray) -> list[np.ndarray]:
    global A_sparse
    A_sparse = sparse.csr_matrix(A)
    bs_list = [bs[i] for i in range(bs.shape[0])]

    xs = []
    with Pool(processes=32) as pool:
        results = pool.imap_unordered(_worker, bs_list, chunksize=1)
        for r in tqdm(results, desc="leap optim", total=len(bs)):
            xs.append(r)

    return xs


def _optimize_compiled_cvxpy_sparse(A: np.ndarray, bs: np.ndarray) -> list[np.ndarray]:
    x = cp.Variable(A.shape[1])
    b_param = cp.Parameter(A.shape[0])
    lambda_reg = 1e-4

    objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b_param) + lambda_reg * cp.sum_squares(x))
    constraints = [x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)

    results = []
    for i in tqdm(range(bs.shape[0]), desc="leap optim"):
        b_param.value = bs[i]
        prob.solve(solver=cp.OSQP, warm_start=True)
        results.append(x.value)

    return results
