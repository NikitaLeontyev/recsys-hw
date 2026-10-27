import numpy as np


class Popularity:
    """Бейзлайн: всем пользователям одинаковый скор = число оценок товара в X."""

    def fit(self, X):
        self.pop = np.asarray(X.sum(axis=0)).ravel()
        return self

    def predict_scores(self, users):
        return np.tile(self.pop, (len(users), 1))