import numpy as np
import scipy.sparse as sp
import torch


class IALS:
    """iALS для неявного фидбэка (Hu, Koren, Volinsky, 2008).

    L = Σ_{u,i} c_ui (p_ui - x_uᵀy_i)² + reg·(‖X‖² + ‖Y‖²),
    p_ui = 1 если есть взаимодействие, иначе 0; c_ui = 1 + alpha для наблюдаемых, 1 для остальных.

    Попеременно фиксируем Y и решаем для каждого пользователя, потом наоборот:
    x_u = (YᵀY + alpha·Σ_{i∈I_u} y_i y_iᵀ + reg·I)⁻¹ · (1 + alpha)·Σ_{i∈I_u} y_i
    YᵀY считается один раз на итерацию, поэтому нули в матрице явно не перебираем.
    """

    def __init__(self, factors=64, reg=1.0, alpha=10.0, iterations=10, seed=42,
                 batch_size=256, device=None):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _batches(self, R):
        """Режем строки R на батчи; сортируем по длине, чтобы паддинг до L был поменьше."""
        lengths = np.diff(R.indptr)
        order = np.argsort(lengths, kind="stable")
        order = order[lengths[order] > 0]
        batches = []
        for start in range(0, len(order), self.batch_size):
            rows = order[start:start + self.batch_size]
            L = lengths[rows].max()
            idx = np.zeros((len(rows), L), dtype=np.int64)
            mask = np.zeros((len(rows), L), dtype=np.float32)
            for k, r in enumerate(rows):
                cols = R.indices[R.indptr[r]:R.indptr[r + 1]]
                idx[k, :len(cols)] = cols
                mask[k, :len(cols)] = 1
            batches.append((torch.from_numpy(rows).to(self.device),
                            torch.from_numpy(idx).to(self.device),
                            torch.from_numpy(mask).to(self.device)))
        return batches

    def _solve(self, batches, n_rows, Y):
        """Один полушаг ALS: решаем линейные системы для всех строк при фиксированном Y."""
        d = Y.shape[1]
        YtY = Y.T @ Y
        eye = torch.eye(d, device=self.device)
        out = torch.zeros((n_rows, d), device=self.device)
        for rows, idx, mask in batches:
            Yu = Y[idx] * mask[..., None]           # (B, L, d), паддинг занулён маской
            A = YtY + self.alpha * (Yu.transpose(1, 2) @ Yu) + self.reg * eye
            b = (1 + self.alpha) * Yu.sum(dim=1)
            out[rows] = torch.linalg.solve(A, b)
        return out

    def fit(self, X):
        self.X = sp.csr_matrix(X, dtype=np.float32)
        XT = self.X.T.tocsr()
        n_users, n_items = self.X.shape
        user_batches = self._batches(self.X)
        item_batches = self._batches(XT)
        torch.manual_seed(self.seed)
        self.U = 0.01 * torch.randn(n_users, self.factors, device=self.device)
        self.V = 0.01 * torch.randn(n_items, self.factors, device=self.device)
        for _ in range(self.iterations):
            self.U = self._solve(user_batches, n_users, self.V)
            self.V = self._solve(item_batches, n_items, self.U)
        return self

    def predict_scores(self, users):
        users = torch.as_tensor(users, device=self.device)
        return (self.U[users] @ self.V.T).cpu().numpy()