import numpy as np
import scipy.sparse as sp
import torch

class EASE:
    """EASE (Steck, 2019): B = argmin ||X - XB||² + reg·||B||², diag(B) = 0.
    Решение в закрытом виде: P = (XᵀX + reg·I)⁻¹, B_ij = -P_ij / P_jj.
    Обращаем матрицу n_items × n_items, на GPU если есть, иначе на CPU."""

    def __init__(self, reg=500.0, device=None):
        self.reg = reg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X):
        self.X = sp.csr_matrix(X, dtype=np.float32)
        # from_numpy не копирует массив, на CPU это экономит ~1.5 ГБ
        G = torch.from_numpy((self.X.T @ self.X).toarray()).to(self.device)
        G.diagonal().add_(self.reg)
        P = torch.linalg.inv(G)
        del G
        P /= -P.diagonal().clone()
        P.fill_diagonal_(0)
        self.B = P.cpu().numpy()
        return self

    def predict_scores(self, users):
        return self.X[users] @ self.B