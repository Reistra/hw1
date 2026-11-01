"""Item-based neighborhood collaborative filtering."""
import numpy as np
from scipy.sparse import csr_matrix


class ItemBasedCF:
    """Cosine item similarity with optional shrinkage and top-neighbor pruning.

    Parameters
    ----------
    top_neighbors : int
        Number of outgoing neighbors retained per item.
    shrink : float
        Shrinkage coefficient applied as cooc / (cooc + shrink).
    """
    def __init__(self, top_neighbors=100, shrink=10.0):
        if top_neighbors <= 0:
            raise ValueError("top_neighbors must be positive")
        if shrink < 0:
            raise ValueError("shrink must be non-negative")
        self.top_neighbors = int(top_neighbors)
        self.shrink = float(shrink)

    def fit(self, X):
        X = csr_matrix(X, dtype=np.float32)
        cooc = (X.T @ X).toarray().astype(np.float32)
        norms = np.sqrt(np.diag(cooc))
        denom = np.outer(norms, norms)
        sim = np.divide(cooc, denom, out=np.zeros_like(cooc), where=denom > 0)
        shrink_denom = cooc + self.shrink
        factor = np.divide(cooc, shrink_denom, out=np.zeros_like(cooc), where=shrink_denom > 0)
        sim *= factor
        np.fill_diagonal(sim, 0.0)
        n_items = sim.shape[0]
        if self.top_neighbors < n_items:
            keep = np.argpartition(sim, -self.top_neighbors, axis=1)[:, -self.top_neighbors:]
            mask = np.zeros_like(sim, dtype=bool)
            mask[np.arange(n_items)[:, None], keep] = True
            sim[~mask] = 0.0
        self.X = X
        self.similarity = csr_matrix(sim)
        return self

    def scores(self, user_idx):
        """Return scores over the item columns in the fitted matrix."""
        return np.asarray(self.X[user_idx] @ self.similarity).ravel()

    def recommend(self, user_idx, k=10, seen_mask=None):
        scores = self.scores(user_idx)
        if seen_mask is not None:
            scores[np.asarray(seen_mask, dtype=bool)] = -np.inf
        eligible = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        top = eligible[np.argsort(scores[eligible])[-k:][::-1]]
        return top.astype(np.int32)
