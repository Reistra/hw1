"""Three-field factorization machine for user, item, and item-category IDs."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FactorizationMachine(nn.Module):
    """Second-order FM specialized to one user, item, and category feature."""
    def __init__(self, n_users, n_items, n_categories, factors=16):
        super().__init__()
        self.linear_user = nn.Embedding(n_users, 1)
        self.linear_item = nn.Embedding(n_items, 1)
        self.linear_category = nn.Embedding(n_categories, 1)
        self.user_f = nn.Embedding(n_users, factors)
        self.item_f = nn.Embedding(n_items, factors)
        self.category_f = nn.Embedding(n_categories, factors)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, user, item, category):
        uf, itf, cf = self.user_f(user), self.item_f(item), self.category_f(category)
        linear = self.linear_user(user) + self.linear_item(item) + self.linear_category(category) + self.bias
        summed = uf + itf + cf
        interaction = .5 * (summed.square() - uf.square() - itf.square() - cf.square()).sum(dim=1, keepdim=True)
        return (linear + interaction).squeeze(1)

    def fit_pairs(self, users, items, categories, labels, *, n_users, n_items,
                  n_categories, factors=16, lr=0.003, weight_decay=1e-5,
                  epochs=3, batch_size=8192, seed=42, device=None):
        """Train from encoded pair arrays and binary labels using BCE-with-logits."""
        torch.manual_seed(seed)
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.__init__(n_users, n_items, n_categories, factors)
        self.to(device)
        tensors = [torch.as_tensor(x, dtype=dtype) for x, dtype in zip(
            (users, items, categories, labels), (torch.long, torch.long, torch.long, torch.float32))]
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True, generator=generator)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        self.device_ = device
        self.train()
        for _ in range(epochs):
            for u, i, c, y in loader:
                u, i, c, y = u.to(device), i.to(device), c.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(self(u, i, c), y)
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def scores(self, user_idx, item_indices, category_indices):
        """Return logits for one user and aligned candidate item/category IDs."""
        device = getattr(self, "device_", next(self.parameters()).device)
        items = torch.as_tensor(item_indices, dtype=torch.long, device=device)
        cats = torch.as_tensor(category_indices, dtype=torch.long, device=device)
        users = torch.full_like(items, int(user_idx))
        self.eval()
        return self(users, items, cats).cpu().numpy()


def sample_negative_pairs(n_users, n_items, positive_pairs, n_negative=None, seed=42):
    """Uniform negative sampling excluding supplied encoded positive (u, i) pairs."""
    rng = np.random.default_rng(seed)
    positives = set(map(tuple, np.asarray(positive_pairs, dtype=np.int64).reshape(-1, 2)))
    target = len(positives) if n_negative is None else int(n_negative)
    negatives, chosen = [], set(positives)
    while len(negatives) < target:
        u, i = int(rng.integers(n_users)), int(rng.integers(n_items))
        if (u, i) not in chosen:
            negatives.append((u, i))
            chosen.add((u, i))
    return np.asarray(negatives, dtype=np.int64).reshape(-1, 2)
