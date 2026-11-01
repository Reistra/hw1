"""EASE: embarrassingly shallow autoencoder for sparse item recommendations."""
import numpy as np
from scipy.sparse import csr_matrix


class EASE:
    def __init__(self, reg=500.0):
        if reg <= 0:
            raise ValueError("reg must be positive")
        self.reg = float(reg)

    def fit(self, X):
        X = csr_matrix(X, dtype=np.float32)
        gram = (X.T @ X).toarray().astype(np.float64)
        gram.flat[::gram.shape[0] + 1] += self.reg
        precision = np.linalg.inv(gram)
        B = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        self.X = X
        return self

    def scores(self, user_idx):
        """Return item scores for a row index in the fitted user-item matrix."""
        return np.asarray(self.X[int(user_idx)] @ self.B).ravel()
