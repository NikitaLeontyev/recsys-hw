import numpy as np
import scipy.sparse as sp
import torch

from .ials import IALS


def build_target(df, user2idx, item2idx, gamma=0.25, half_life_days=180.0,
                 floor=0.2, beta=5.0, min_pref=0.05):
    """Целевой сигнал p_ui = pref_ui * decay_ui.

    pref_ui  = max(min_pref, 1 + gamma * (r_ui - r̄_u)),
               r̄_u = (Σ r_u + beta * μ) / (n_u + beta)   — сглаженное среднее пользователя
    decay_ui = floor + (1 - floor) * 0.5 ** ((t_max - t_ui) / half_life)

    t_max — последний момент обучающих данных (никакой информации из будущего).
    gamma=0 и floor=1 дают p_ui = 1, то есть обычный iALS.
    df должен быть без дублей (user, item), иначе csr сложит значения.
    """
    mu = df["rating"].mean()
    g = df.groupby("user_id")["rating"]
    r_bar = (g.transform("sum") + beta * mu) / (g.transform("count") + beta)
    pref = np.maximum(min_pref, 1.0 + gamma * (df["rating"] - r_bar)).values

    t_days = df["timestamp"].values / 86_400_000.0  # Amazon 2023: timestamp в мс
    dt = t_days.max() - t_days
    decay = floor + (1.0 - floor) * 0.5 ** (dt / half_life_days)

    p = (pref * decay).astype(np.float32)
    rows = df["user_id"].map(user2idx).values
    cols = df["parent_asin"].map(item2idx).values
    return sp.csr_matrix((p, (rows, cols)), shape=(len(user2idx), len(item2idx)))


class IALSCustom(IALS):
    """iALS с вещественным таргетом p_ui вместо бинарного 1.

    L = Σ_{(u,i)∈obs} c (p_ui - x_uᵀy_i)² + Σ_{(u,i)∉obs} (x_uᵀy_i)² + λ(‖X‖² + ‖Y‖²),
    c = 1 + α — одинаковая для всех наблюдений, меняется только целевое значение.

    x_u = (YᵀY + α Σ_i y_i y_iᵀ + λI)⁻¹ · (1+α) Σ_i p_ui y_i

    fit(P) принимает матрицу из build_target: ненулевые элементы — наблюдения,
    значения — таргет. При P = бинарной X совпадает с IALS.
    """

    def _batches(self, R):
        lengths = np.diff(R.indptr)
        order = np.argsort(lengths, kind="stable")
        order = order[lengths[order] > 0]
        batches = []
        for start in range(0, len(order), self.batch_size):
            rows = order[start:start + self.batch_size]
            L = lengths[rows].max()
            idx = np.zeros((len(rows), L), dtype=np.int64)
            mask = np.zeros((len(rows), L), dtype=np.float32)
            val = np.zeros((len(rows), L), dtype=np.float32)
            for k, r in enumerate(rows):
                s, e = R.indptr[r], R.indptr[r + 1]
                idx[k, :e - s] = R.indices[s:e]
                mask[k, :e - s] = 1
                val[k, :e - s] = R.data[s:e]
            batches.append(tuple(torch.from_numpy(a).to(self.device)
                                 for a in (rows, idx, mask, val)))
        return batches

    def _solve(self, batches, n_rows, Y):
        d = Y.shape[1]
        YtY = Y.T @ Y
        eye = torch.eye(d, device=self.device)
        out = torch.zeros((n_rows, d), device=self.device)
        for rows, idx, mask, val in batches:
            Yu = Y[idx] * mask[..., None]
            A = YtY + self.alpha * (Yu.transpose(1, 2) @ Yu) + self.reg * eye
            b = (1 + self.alpha) * (Yu * val[..., None]).sum(dim=1)
            out[rows] = torch.linalg.solve(A, b)
        return out