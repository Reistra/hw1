"""Alternating least squares for implicit feedback (Hu, Koren, Volinsky form)."""
import numpy as np
from scipy.sparse import csr_matrix


class ImplicitALS:
    def __init__(self, factors=16, alpha=20.0, reg=0.1, iterations=8, seed=42):
        if factors <= 0 or iterations <= 0 or alpha < 0 or reg <= 0:
            raise ValueError("Require factors/iterations > 0, alpha >= 0, reg > 0")
        self.factors, self.alpha, self.reg = int(factors), float(alpha), float(reg)
        self.iterations, self.seed = int(iterations), int(seed)

    def fit(self, X):
        X = csr_matrix(X, dtype=np.float32)
        self.active_users = np.flatnonzero(np.asarray(X.getnnz(axis=1)).ravel() > 0)
        self.X = X[self.active_users].tocsr()
        self.user_to_local = {int(u): j for j, u in enumerate(self.active_users)}
        rng = np.random.default_rng(self.seed)
        n_users, n_items = self.X.shape
        self.U = rng.normal(0, .01, (n_users, self.factors)).astype(np.float32)
        self.V = rng.normal(0, .01, (n_items, self.factors)).astype(np.float32)
        eye = np.eye(self.factors, dtype=np.float32)
        for _ in range(self.iterations):
            yty = self.V.T @ self.V
            for u in range(n_users):
                a, b = self.X.indptr[u:u+2]
                idx, vals = self.X.indices[a:b], self.X.data[a:b]
                if not len(idx):
                    continue
                y = self.V[idx]
                cm1 = self.alpha * vals
                A = yty + (y.T * cm1) @ y + self.reg * eye
                rhs = y.T @ (1.0 + cm1)
                self.U[u] = np.linalg.solve(A, rhs)
            xtx = self.U.T @ self.U
            Xcsc = self.X.tocsc()
            for i in range(n_items):
                a, b = Xcsc.indptr[i:i+2]
                idx, vals = Xcsc.indices[a:b], Xcsc.data[a:b]
                if not len(idx):
                    continue
                x = self.U[idx]
                cm1 = self.alpha * vals
                A = xtx + (x.T * cm1) @ x + self.reg * eye
                rhs = x.T @ (1.0 + cm1)
                self.V[i] = np.linalg.solve(A, rhs)
        return self

    def scores(self, user_idx):
        """Return scores over all fitted item columns; unknown users get zeros."""
        local = self.user_to_local.get(int(user_idx))
        if local is None:
            return np.zeros(self.V.shape[0], dtype=np.float32)
        return self.U[local] @ self.V.T
