from typing import Literal

import numpy as np
import quapy as qp
import torch
import torch.nn as nn
import torch.optim as optim
from quapy.functional import prevalence_from_labels
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from method._gaussian_kde_cuda import KernelDensityCuda


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        n_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class KDEyMLCuda:
    def __init__(
        self,
        classifier_params: dict = None,
        bandwidth: float = 0.1,
        cv_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        kde_batch_size: int = 1000,
        max_optim_iter: int = 500,
        random_state: int = None,
        return_type: Literal["np", "pt"] = "np",
    ):
        self.classifier_params = classifier_params or {"hidden_dims": [128], "dropout": 0.3}
        self.bandwidth = bandwidth
        self.cv_folds = cv_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.kde_batch_size = kde_batch_size
        self.max_optim_iter = max_optim_iter
        self.random_state = random_state if random_state is not None else qp.environ["_R_SEED"]
        self.return_type = return_type

        self.optimizer_type = "lbfgs"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
        self.kdes = None
        self.classes_ = None
        self.n_classes = None

        # TODO: fix with context manager
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

    def _train_model(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor, pbar: tqdm):
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        best_loss = float("inf")
        last_best = 0

        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            if avg_loss < best_loss:
                if (best_loss - avg_loss) >= 1e-4:
                    last_best = epoch
                best_loss = avg_loss

            if (epoch - last_best) >= 10:
                pbar.update(self.epochs - epoch)
                break
            pbar.update(1)

    def _train_classifier_cv(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        n_samples, n_features = X.shape
        posteriors = torch.zeros((n_samples, self.n_classes), device=self.device)

        pbar = tqdm(desc="Training", total=(self.cv_folds + 1) * self.epochs)

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in skf.split(torch.zeros(len(y)), y.cpu().numpy()):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]

            model = MLP(
                input_dim=n_features,
                n_classes=self.n_classes,
                **self.classifier_params,
            ).to(self.device)

            self._train_model(model, X_train_fold, y_train_fold, pbar=pbar)

            model.eval()
            with torch.no_grad():
                X_val_fold = X_val_fold.to(self.device)

                fold_posteriors = []
                for i in range(0, len(X_val_fold), self.batch_size):
                    batch = X_val_fold[i : i + self.batch_size]
                    outputs = model(batch)
                    probs = torch.softmax(outputs, dim=1)
                    fold_posteriors.append(probs)

                posteriors[val_idx] = torch.cat(fold_posteriors)

        self.classifier = MLP(
            input_dim=n_features,
            n_classes=self.n_classes,
            **self.classifier_params,
        ).to(self.device)

        self._train_model(self.classifier, X, y, pbar=pbar)

        return posteriors

    def fit(self, X: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor):
        X_tensor = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X
        y_tensor = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y

        self.classes_ = torch.unique(y_tensor)
        self.n_classes = len(self.classes_)

        # INFO: train the classifier and obtain posteriors on cv folds
        posteriors = self._train_classifier_cv(X_tensor, y_tensor)

        # INFO: build the kdes
        self.kdes: list[KernelDensityCuda] = []
        for c in self.classes_:
            class_mask = y_tensor == c
            class_posteriors = posteriors[class_mask]
            kde = KernelDensityCuda(
                bandwidth=self.bandwidth,
                test_batch_size=self.kde_batch_size,
                return_type="pt",
            )
            kde.fit(class_posteriors)
            self.kdes.append(kde)

        return self

    def quantify(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        X_tensor = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X

        self.classifier.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)

            all_posteriors = []
            for i in range(0, len(X_tensor), self.batch_size):
                batch = X_tensor[i : i + self.batch_size]
                outputs = self.classifier(batch)
                probs = torch.softmax(outputs, dim=1)
                all_posteriors.append(probs)

            posteriors = torch.cat(all_posteriors, dim=0)

        log_densities = torch.zeros((len(X), self.n_classes), device=self.device)
        for c in range(self.n_classes):
            log_densities[:, c] = self.kdes[c].score_samples(posteriors)

        prevalences = self._optimize_mixture(log_densities)

        if self.return_type == "np":
            prevalences = prevalences.cpu().numpy()

        return prevalences

    def batch_quantify(self, Xs: list[np.ndarray] | list[torch.Tensor]) -> np.ndarray | torch.Tensor:
        n_batches = len(Xs)
        if isinstance(Xs[0], np.ndarray):
            Xs = torch.from_numpy(np.stack(Xs, axis=0)).float()
        else:
            Xs = torch.stack(Xs, dim=0)

        Xs = Xs.view(-1, Xs.size(2))

        self.classifier.eval()
        with torch.no_grad():
            Xs_tensor = Xs.to(self.device)

            all_posteriors = []
            for i in tqdm(range(0, len(Xs_tensor), self.batch_size), desc="Posteriors"):
                batch = Xs_tensor[i : i + self.batch_size]
                outputs = self.classifier(batch)
                probs = torch.softmax(outputs, dim=1)
                all_posteriors.append(probs)

            posteriors = torch.cat(all_posteriors, dim=0)

        log_densities = torch.zeros((len(Xs), self.n_classes), device=self.device)
        for c in tqdm(range(self.n_classes), desc="KDE"):
            log_densities[:, c] = self.kdes[c].score_samples(posteriors)

        log_densities = log_densities.view(n_batches, -1, self.n_classes)
        prevalences = self._batch_optimize_mixture(log_densities)

        if self.return_type == "np":
            prevalences = prevalences.cpu().numpy()

        return prevalences

    def _batch_optimize_mixture(self, log_densities: torch.Tensor) -> torch.Tensor:
        n_batches = log_densities.size(0)
        all_prevalences = []

        for b in tqdm(range(n_batches), desc="q optim"):
            b_prevalences = self._optimize_mixture(log_densities[b])
            all_prevalences.append(b_prevalences)

        prevalences = torch.stack(all_prevalences, dim=0)

        return prevalences

    # def _optimize_mixture(self, log_densities: torch.Tensor) -> torch.Tensor:
    #     logits = torch.zeros(self.n_classes, device=self.device, requires_grad=True)
    #     optimizer = optim.LBFGS([logits], lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    #
    #     iteration = [0]
    #     best_loss = [float("inf")]
    #     best_prevalences = [None]
    #
    #     def closure():
    #         optimizer.zero_grad()
    #         prevalences = torch.softmax(logits, dim=0)
    #         log_prevalences = torch.log(prevalences + 1e-10)
    #         log_weighted_densities = log_densities + log_prevalences
    #
    #         max_log, _ = log_weighted_densities.max(dim=1, keepdim=True)
    #         log_mixture = max_log + torch.log(torch.exp(log_weighted_densities - max_log).sum(dim=1, keepdim=True))
    #
    #         nll = -log_mixture.sum()
    #         nll.backward()
    #
    #         current_loss = nll.item()
    #         if current_loss < best_loss[0]:
    #             best_loss[0] = current_loss
    #             best_prevalences[0] = torch.softmax(logits.detach().clone(), dim=0)
    #
    #         iteration[0] += 1
    #         return nll
    #
    #     for _ in range(self.max_optim_iter // 20):
    #         optimizer.step(closure)
    #         if iteration[0] >= self.max_optim_iter:
    #             break
    #
    #     prevalences = best_prevalences[0] if best_prevalences[0] is not None else torch.softmax(logits.detach(), dim=0)
    #     return prevalences

    def _optimize_mixture(self, log_densities: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(self.n_classes, device=self.device, requires_grad=True)
        optimizer = optim.Adam([logits], lr=0.2)

        best_loss = float("inf")
        best_prevalences = None
        tol = 1e-6
        patience = 10
        no_improve = 0

        for _ in range(100):
            optimizer.zero_grad()

            prevalences = torch.softmax(logits, dim=0)
            log_prevalences = torch.log(prevalences + 1e-10)

            log_weighted = log_densities + log_prevalences
            max_log = log_weighted.max(dim=1, keepdim=True)[0]
            log_mixture = max_log + torch.log(torch.exp(log_weighted - max_log).sum(dim=1, keepdim=True))

            nll = -log_mixture.sum()
            nll.backward()
            optimizer.step()

            current_loss = nll.item()
            if current_loss < best_loss - tol:
                best_loss = current_loss
                best_prevalences = prevalences.detach().clone()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        prevalences = best_prevalences if best_prevalences is not None else torch.softmax(logits.detach(), dim=0)
        return prevalences / prevalences.sum()
