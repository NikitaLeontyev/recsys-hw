import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def load_meta(path, item2idx):
    """Метаданные товаров из item2idx. average_rating, rating_number, bought_together
    не берём: это агрегаты на момент сбора датасета (2023), т.е. информация из будущего."""
    keep = set(item2idx)
    parts = [ch.loc[ch["parent_asin"].isin(keep), ["parent_asin", "store", "categories"]]
             for ch in pd.read_json(path, lines=True, chunksize=100_000)]
    return pd.concat(parts, ignore_index=True)


def build_item_features(meta, item2idx, use_meta=True, min_count=2, meta_weight=1.0):
    """Признаки товара: item_id + жанры + артист + формат (CD/винил) из поля store.
    Редкие значения (< min_count товаров) отбрасываем.
    Вес item_id = 1, вес каждого метапризнака = meta_weight / sqrt(число метапризнаков),
    чтобы метаданные уточняли товар, а не заглушали его собственный вектор.
    Возвращает padded-матрицы индексов признаков и весов (индекс 0 — паддинг)."""
    tokens = [[] for _ in range(len(item2idx))]
    if use_meta:
        for asin, store, cats in zip(meta["parent_asin"], meta["store"], meta["categories"]):
            if asin not in item2idx:
                continue
            t = ["genre=" + c for c in (cats if isinstance(cats, list) else []) if c != "CDs & Vinyl"]
            artist, _, fmt = (store if isinstance(store, str) else "").partition("Format:")
            if artist.strip():
                t.append("artist=" + artist.strip().lower())
            if fmt.strip():
                t.append("format=" + fmt.strip().lower())
            tokens[item2idx[asin]] = t

    cnt = pd.Series([t for ts in tokens for t in set(ts)]).value_counts()
    vocab = {}
    lists = []
    for i, ts in enumerate(tokens):
        ts = [f"item={i}"] + [t for t in ts if cnt[t] >= min_count]
        lists.append([vocab.setdefault(t, len(vocab) + 1) for t in ts])

    L = max(len(l) for l in lists)
    idx = np.zeros((len(lists), L), dtype=np.int64)
    wt = np.zeros((len(lists), L), dtype=np.float32)
    for k, l in enumerate(lists):
        idx[k, :len(l)] = l
        wt[k, 0] = 1.0                                   # item_id
        if len(l) > 1:                                   # метапризнаки: meta_weight / sqrt(n)
            wt[k, 1:len(l)] = meta_weight / np.sqrt(len(l) - 1)
    return idx, wt, len(vocab) + 1


class FM:
    """Факторизационная машина: пользователь = user_id, товар = набор признаков.

    ŷ(u, i) = Σ_{f∈i} w_f + ⟨v_u, s_i⟩ + ½(‖s_i‖² − Σ_{f∈i} ‖v_f‖²),   s_i = Σ_{f∈i} v_f
    (стандартная FM 2-го порядка; слагаемые, зависящие только от u, сокращаются при ранжировании).

    Обучение BPR: −log σ(ŷ(u,i⁺) − ŷ(u,i⁻)) + reg·(‖v_u‖² + ‖s_i⁺‖² + ‖s_i⁻‖²).
    Негатив i⁻ сэмплируется ∝ популярность^neg_power: при neg_power=0 — равномерно,
    при >0 модель учится отличать купленное от популярного, а не от случайного.
    item_inter=False убирает попарные взаимодействия признаков внутри товара (½(...)):
    они не зависят от пользователя и работают как «популярность сочетания признаков»;
    остаются только взаимодействия пользователь × признак товара."""

    def __init__(self, item_feats, factors=64, lr=3e-3, reg=1e-4, epochs=20,
                 neg_power=0.0, item_inter=True, batch_size=4096, seed=42, device=None):
        self.item_idx, self.item_wt, self.n_features = item_feats
        self.item_inter = item_inter
        self.factors, self.lr, self.reg, self.epochs = factors, lr, reg, epochs
        self.neg_power, self.batch_size, self.seed = neg_power, batch_size, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _item(self, items):
        idx, wt = self.I[items], self.Wt[items]   # (B, L) индексы и веса признаков
        v = self.V[idx] * wt[..., None]
        s = v.sum(1)
        bias = (self.w[idx] * wt).sum(1)
        if self.item_inter:
            bias = bias + 0.5 * ((s * s).sum(1) - (v * v).sum((1, 2)))
        return s, bias

    def fit(self, X):
        X = sp.csr_matrix(X)
        dev = self.device
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_users, n_items = X.shape
        self.I = torch.from_numpy(self.item_idx).to(dev)
        self.Wt = torch.from_numpy(self.item_wt).to(dev)
        self.U = (0.01 * torch.randn(n_users, self.factors, device=dev)).requires_grad_()
        self.V = (0.01 * torch.randn(self.n_features, self.factors, device=dev)).requires_grad_()
        self.w = torch.zeros(self.n_features, device=dev, requires_grad=True)
        opt = torch.optim.Adam([self.U, self.V, self.w], lr=self.lr)

        pop = np.asarray(X.sum(axis=0)).ravel() + 1
        neg_p = torch.tensor(pop ** self.neg_power, dtype=torch.float32, device=dev)
        users, items = X.nonzero()
        users, items = users.astype(np.int64), items.astype(np.int64)

        for _ in range(self.epochs):
            perm = rng.permutation(len(users))
            for st in range(0, len(perm), self.batch_size):
                b = perm[st:st + self.batch_size]
                u = torch.from_numpy(users[b]).to(dev)
                ip = torch.from_numpy(items[b]).to(dev)
                ineg = torch.multinomial(neg_p, len(b), replacement=True)
                vu = self.U[u]
                sp_, bp = self._item(ip)
                sn, bn = self._item(ineg)
                pos = bp + (vu * sp_).sum(1)
                neg = bn + (vu * sn).sum(1)
                loss = -F.logsigmoid(pos - neg).mean() + self.reg * (
                    vu.pow(2).sum(1) + sp_.pow(2).sum(1) + sn.pow(2).sum(1)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        with torch.no_grad():
            self.S, self.b = self._item(torch.arange(n_items, device=dev))
        return self

    def predict_scores(self, users):
        with torch.no_grad():
            u = torch.as_tensor(users, dtype=torch.long, device=self.device)
            return (self.U[u] @ self.S.T + self.b).cpu().numpy()