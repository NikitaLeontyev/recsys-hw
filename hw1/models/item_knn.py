import numpy as np
import scipy.sparse as sp


class ItemKNN:
    """Item-based kNN.
    sim(i, j) — косинус между столбцами X (векторами товаров по пользователям).
    У каждого товара оставляем k самых похожих соседей, остальные сходства = 0.
    score(u, i) = сумма sim(i, j) по товарам j из истории u, у которых i среди соседей.
    """

    def __init__(self, k=100, block_size=1000):
        self.k = k
        self.block_size = block_size

    def fit(self, X):
        self.X = sp.csr_matrix(X, dtype=np.float32)
        n_items = self.X.shape[1]
        norms = np.sqrt(np.asarray(self.X.power(2).sum(axis=0)).ravel())
        norms[norms == 0] = 1
        Xn = (self.X @ sp.diags(1 / norms)).tocsc()
        XnT = Xn.T.tocsr()

        # считаем X^T X блоками по block_size товаров, чтобы не держать в памяти полную матрицу
        rows, cols, vals = [], [], []
        for start in range(0, n_items, self.block_size):
            end = min(start + self.block_size, n_items)
            block = (XnT[start:end] @ Xn).toarray()
            block[np.arange(end - start), np.arange(start, end)] = 0      # сам с собой не сосед
            top = np.argpartition(-block, self.k, axis=1)[:, :self.k]
            rows.append(np.repeat(np.arange(start, end), self.k))
            cols.append(top.ravel())
            vals.append(np.take_along_axis(block, top, axis=1).ravel())

        S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n_items, n_items))
        S.eliminate_zeros()
        self.W = S.T.tocsr()   # W[j, i] = sim(i, j), если j в топ-k соседей i
        return self

    def predict_scores(self, users):
        return (self.X[users] @ self.W).toarray()