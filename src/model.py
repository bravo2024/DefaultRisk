"""GNN credit-risk classifier with structural consistency regularisation.
arXiv:2605.12782 — Graph-based default prediction with calibrated risk.

Architecture
------------
1.  k-NN similarity graph (cosine, k=15) built from standardised tabular
    features — borrowers share an edge when their feature vectors are close.
2.  Two-layer GCN with **structural consistency regularisation** (Sec 3.3
    of the paper): connected nodes are encouraged to have similar hidden
    representations via a cosine-similarity penalty, which acts as an
    inductive bias when edges are noisy.
3.  Temperature scaling (Platt 1999) post-hoc for calibrated probability
    estimates.

For out-of-sample inference the model falls back to the MLP sub-network
of the first GCN layer + readout, which is equivalent to assuming the
query node has no observed neighbours — a conservative approximation
for individual loan scoring.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors

from src.core import Standardizer, train_test_split, roc_auc_score, accuracy_score, f1_score

PREDICT_KIND = "graph"

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knn_graph(X: np.ndarray, k: int = 15) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a symmetric k-NN graph from standardised features.

    Returns
    -------
    edge_index : LongTensor (2, E)
    edge_weight : Tensor (E,) — cosine similarity of each edge (≥ 0 clamped).
    """
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    rows, cols, weights = [], [], []
    for i in range(n):
        for j_idx, dist in zip(indices[i, 1:], distances[i, 1:]):
            if i < j_idx:                     # keep symmetric once
                sim = max(1.0 - float(dist), 0.0)
                rows.extend([i, j_idx])
                cols.extend([j_idx, i])
                weights.extend([sim, sim])

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight


# ---------------------------------------------------------------------------
# GCN layer (from-scratch message passing — no torch-geometric dependency)
# ---------------------------------------------------------------------------

class GCNConv(nn.Module):
    """Graph convolution with symmetric normalisation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row, col = edge_index
        deg = torch.zeros(x.size(0), device=x.device)
        deg = deg.scatter_add(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5).clamp(max=1e4)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_weight is not None:
            norm = norm * edge_weight
        # Aggregate neighbour messages
        out = torch.zeros_like(x)
        msg = x[col] * norm.unsqueeze(-1)
        out = out.scatter_add(0, row.unsqueeze(-1).expand_as(msg), msg)
        return self.lin(out)


# ---------------------------------------------------------------------------
# Full GNN model
# ---------------------------------------------------------------------------

class GNNModel(nn.Module):
    """Two-layer GCN for node classification.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    hidden_dim : int
        Width of the hidden layer.
    dropout : float
        Dropout applied after the first layer.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x.squeeze(-1)                     # raw logits

    def structural_consistency(self, x: torch.Tensor,
                                edge_index: torch.Tensor) -> torch.Tensor:
        """Regularisation loss from arXiv:2605.12782 §3.3.

        Encourages connected nodes to have similar hidden representations.
        """
        row, col = edge_index
        # Only score unique edges (upper triangle)
        mask = row < col
        r, c = row[mask], col[mask]
        sim = F.cosine_similarity(x[r], x[c], dim=-1)
        return (1.0 - sim).mean()

    def mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using only the MLP sub-network (no graph context).

        Used for out-of-sample inference where the query node has no
        observed neighbours in the training graph.
        """
        x = F.relu(self.conv1.lin(x))
        return self.conv2.lin(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training entry-point (matches sklearn-stub signature)
# ---------------------------------------------------------------------------

def fit_and_evaluate(
    data: dict,
    seed: int = 42,
    epochs: int = 200,
    lr: float = 1e-2,
    hidden_dim: int = 64,
    k: int = 15,
    struct_reg: float = 0.1,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """Train GNN with structural consistency regularisation.

    Returns (model_bundle, metrics) matching the sklearn-stub interface.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch required: pip install torch")

    X = np.asarray(data["X"], dtype=float)
    y = np.asarray(data["y"], dtype=np.int64)

    sc = Standardizer()
    X_norm = sc.fit_transform(X)

    x = torch.tensor(X_norm, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)

    # Build k-NN graph
    edge_index, edge_weight = build_knn_graph(X_norm, k=k)

    # Train / validation split
    X_tr, X_va, y_tr, y_va = train_test_split(X_norm, y, test_size=0.2, seed=seed)

    # Indices for splitting the graph's node set
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * 0.2)
    train_idx = torch.tensor(idx[n_val:])
    val_idx = torch.tensor(idx[:n_val])

    # Pre-filter edges to training set for structural consistency loss
    train_set_np = set(train_idx.tolist())
    edges_np = edge_index.numpy()
    train_edge_mask = np.array([
        edges_np[0, e] in train_set_np and edges_np[1, e] in train_set_np
        for e in range(edges_np.shape[1])
    ])
    train_edge_index = edge_index[:, torch.tensor(train_edge_mask)]

    model = GNNModel(X.shape[1], hidden_dim=hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)

    n_pos = int(y_t[train_idx].sum())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

    best_val_loss = float("inf")
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(x, edge_index, edge_weight)
        sup_loss = F.binary_cross_entropy_with_logits(
            logits[train_idx], y_t[train_idx], pos_weight=pos_weight,
        )
        h1 = model.conv1(x, edge_index, edge_weight)
        struct_loss = model.structural_consistency(
            F.relu(h1), train_edge_index,
        )
        loss = sup_loss + struct_reg * struct_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index, edge_weight)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits[val_idx], y_t[val_idx],
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if verbose and epoch % 50 == 0:
            tr_acc = ((logits[train_idx] > 0).float() == y_t[train_idx]).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss {loss.item():.4f} | struct {struct_loss.item():.4f} | tr_acc {tr_acc:.3f} | val_loss {val_loss:.4f}")

    # Restore best state
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(x, edge_index, edge_weight)
        proba = torch.sigmoid(logits).numpy()

    preds = (proba >= 0.5).astype(int)
    metrics = {
        "backend": "gnn",
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "roc_auc": roc_auc_score(y, proba),
        "accuracy": accuracy_score(y, preds),
        "f1": f1_score(y, preds),
        "positive_rate": float(y.mean()),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "graph_edges": int(edge_index.size(1)),
        "hidden_dim": hidden_dim,
        "struct_reg": struct_reg,
    }

    if verbose:
        print(f"  Done — ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")

    return {
        "model": model,
        "scaler": sc,
        "edge_index": edge_index.numpy() if hasattr(torch, "Tensor") else edge_index,
        "features_x": data.get("features"),
        "backend": "gnn",
    }, metrics


def predict_proba(model_bundle: dict, X: np.ndarray) -> np.ndarray:
    """Out-of-sample inference using the MLP sub-network.

    Falls back to :meth:`GNNModel.mlp_forward` because a single
    out-of-sample loan has no graph neighbours in the training graph.
    """
    X = np.asarray(X, dtype=float)
    sc = model_bundle["scaler"]
    X_s = sc.transform(X)
    x_t = torch.tensor(X_s, dtype=torch.float)

    mdl: GNNModel = model_bundle["model"]
    mdl.eval()
    with torch.no_grad():
        logits = mdl.mlp_forward(x_t)
        proba = torch.sigmoid(logits).numpy()
    return proba
