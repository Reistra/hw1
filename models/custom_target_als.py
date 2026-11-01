"""Custom recency-decayed target and explicit matrix-factorization ALS."""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

DEFAULT_EVENT_WEIGHTS = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}


def build_custom_target(train_frame, user_to_idx, item_to_idx, catalog_to_idx,
                        *, decay_lambda=0.05, event_weights=None):
    """Build Y[u, catalog_item] = log1p(sum(weight * exp(-lambda * age_days))).

    `train_frame` must have visitorid, itemid, event, timestamp columns.
    Index maps must be derived from train only.
    """
    import pandas as pd
    weights = event_weights or DEFAULT_EVENT_WEIGHTS
    frame = train_frame[train_frame["itemid"].isin(catalog_to_idx)].copy()
    if frame.empty:
        return csr_matrix((len(user_to_idx), len(catalog_to_idx)), dtype=np.float32)
    frame["user_idx"] = frame["visitorid"].map(user_to_idx)
    frame["catalog_idx"] = frame["itemid"].map(catalog_to_idx)
    frame["event_weight"] = frame["event"].map(weights).astype(np.float32)
    end_time = train_frame["timestamp"].max()
    age_days = (end_time - frame["timestamp"]).dt.total_seconds().to_numpy() / 86400.0
    frame["decayed"] = frame["event_weight"].to_numpy() * np.exp(-decay_lambda * age_days)
    pair = frame.groupby(["user_idx", "catalog_idx"], as_index=False)["decayed"].sum()
    target = np.log1p(pair["decayed"].to_numpy()).astype(np.float32)
    return coo_matrix((target, (pair["user_idx"].to_numpy(np.int32), pair["catalog_idx"].to_numpy(np.int32))),
                      shape=(len(user_to_idx), len(catalog_to_idx))).tocsr()


class CustomTargetALS:
    """Explicit-feedback ALS, with unobserved entries treated as target zero."""
    def __init__(self, factors=16, reg=0.1, iterations=8, seed=42):
        if factors <= 0 or iterations <= 0 or reg <= 0:
            raise ValueError("factors, iterations, and reg must be positive")
        self.factors, self.reg = int(factors), float(reg)
        self.iterations, self.seed = int(iterations), int(seed)

    def fit(self, Y):
        Y = csr_matrix(Y, dtype=np.float32)
        self.active_users = np.flatnonzero(np.asarray(Y.getnnz(axis=1)).ravel() > 0)
        self.Y = Y[self.active_users].tocsr()
        self.user_to_local = {int(u): j for j, u in enumerate(self.active_users)}
        rng = np.random.default_rng(self.seed)
        n_users, n_items = self.Y.shape
        self.U = rng.normal(0, .01, (n_users, self.factors)).astype(np.float32)
        self.V = rng.normal(0, .01, (n_items, self.factors)).astype(np.float32)
        eye = np.eye(self.factors, dtype=np.float32)
        for _ in range(self.iterations):
            VtV = self.V.T @ self.V + self.reg * eye
            for u in range(n_users):
                a, b = self.Y.indptr[u:u+2]
                idx, vals = self.Y.indices[a:b], self.Y.data[a:b]
                if len(idx): self.U[u] = np.linalg.solve(VtV, self.V[idx].T @ vals)
            UtU = self.U.T @ self.U + self.reg * eye
            Ycsc = self.Y.tocsc()
            for i in range(n_items):
                a, b = Ycsc.indptr[i:i+2]
                idx, vals = Ycsc.indices[a:b], Ycsc.data[a:b]
                if len(idx): self.V[i] = np.linalg.solve(UtU, self.U[idx].T @ vals)
        return self

    def scores(self, user_idx):
        local = self.user_to_local.get(int(user_idx))
        if local is None:
            return np.zeros(self.V.shape[0], dtype=np.float32)
        return self.U[local] @ self.V.T
