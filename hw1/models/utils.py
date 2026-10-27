import pandas as pd
import numpy as np
import scipy.sparse as sp

def k_core(d, k_user, k_item, verbose=True):
    """Итеративно выкидываем товары с < k_item оценок и пользователей с < k_user,
    пока ничего не меняется (после удаления товаров у пользователей могут стать < k оценок)."""
    it = 0
    while True:
        it += 1
        n_before = len(d)

        item_cnt = d["parent_asin"].value_counts()
        d = d[d["parent_asin"].isin(item_cnt[item_cnt >= k_item].index)]

        user_cnt = d["user_id"].value_counts()
        d = d[d["user_id"].isin(user_cnt[user_cnt >= k_user].index)]

        if verbose:
            print(f"Итерация {it}: {len(d):,} строк")
        if len(d) == n_before:
            return d


def build_matrix(train_df):
    """Бинарная матрица user × item по train и словари id -> индекс.
    Индексы строятся только по train, поэтому новых пользователей/товаров в ней нет."""
    users = train_df["user_id"].unique()
    items = train_df["parent_asin"].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    rows = train_df["user_id"].map(user2idx).values
    cols = train_df["parent_asin"].map(item2idx).values
    vals = np.ones(len(train_df), dtype=np.float32)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    return X, user2idx, item2idx


def build_ground_truth(eval_df, user2idx, item2idx, min_rating=4):
    """{user: множество релевантных товаров} для val/test.
    Релевантно = оценка >= min_rating. Пользователи и товары, которых нет в train, выкидываются."""
    d = eval_df[eval_df["rating"] >= min_rating]
    u = d["user_id"].map(user2idx)
    i = d["parent_asin"].map(item2idx)
    mask = u.notna() & i.notna()
    gt = {}
    for uu, ii in zip(u[mask].astype(int), i[mask].astype(int)):
        gt.setdefault(uu, set()).add(ii)
    return gt

def topk_from_scores(scores, X_seen, k):
    """Топ-k по скорам; товары, которые пользователь уже оценил в train (X_seen), исключаются."""
    scores = scores.astype(np.float64)
    scores[X_seen.nonzero()] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, top], axis=1)
    return top[rows, order]

def evaluate(model, X_train, gt, k=10, batch_size=1000):
    """Recall@k = |rec ∩ rel| / |rel|,  NDCG@k = DCG@k / IDCG@k (бинарная релевантность).
    Усредняем по пользователям из gt. Кандидаты — все товары train кроме уже оценённых."""
    users = np.array(sorted(gt.keys()))
    discounts = 1 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        scores = model.predict_scores(batch)
        top = topk_from_scores(scores, X_train[batch], k)
        for u, rec in zip(batch, top):
            rel = gt[u]
            hits = np.isin(rec, list(rel))
            recalls.append(hits.sum() / len(rel))
            idcg = discounts[:min(len(rel), k)].sum()
            ndcgs.append((hits * discounts).sum() / idcg)
    return {f"Recall@{k}": np.mean(recalls), f"NDCG@{k}": np.mean(ndcgs)}

def matrix_with_maps(df, user2idx, item2idx):
    """Как build_matrix, но с уже готовыми словарями (чтобы размеры совпадали с X_tr)."""
    rows = df["user_id"].map(user2idx).values
    cols = df["parent_asin"].map(item2idx).values
    vals = np.ones(len(df), dtype=np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(user2idx), len(item2idx)))