"""
Krum Robust Aggregation (Blanchard et al., 2017)
Byzantine-resilient alternative to FedAvg averaging.

Single-Krum: select the ONE update closest (in L2 score) to its k nearest neighbors.
Multi-Krum:  select the top-m scoring updates, then average them.

Requirement: n > 2f + 2  (n=total clients, f=max malicious)
Complexity:  O(n² × d)   (n=clients, d=model parameters)
"""
from typing import List, Tuple
import numpy as np


def compute_pairwise_distances(updates: List[np.ndarray]) -> np.ndarray:
    """
    Compute n×n matrix of pairwise squared L2 distances between flattened updates.
    updates: list of n flattened gradient vectors, each shape (d,)
    Returns: (n, n) distance matrix
    """
    n = len(updates)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sum((updates[i] - updates[j]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def krum_scores(updates: List[np.ndarray], f: int) -> np.ndarray:
    """
    Compute the Krum score for each update.
    Score(i) = sum of squared distances from i to its k=n-f-2 nearest neighbors.
    Lower score = more "honest-looking" = safer to select.

    Args:
        updates: list of n flattened gradient vectors
        f:       number of assumed Byzantine (malicious) clients
    Returns:
        scores: array of shape (n,) with Krum score per client
    """
    n = len(updates)
    k = n - f - 2   # number of neighbors to consider
    assert k > 0, f"k={k} ≤ 0. Need n > 2f+2. Got n={n}, f={f}"

    dist = compute_pairwise_distances(updates)

    scores = np.zeros(n)
    for i in range(n):
        # sort distances from i to all others, take k closest (excluding self)
        row = dist[i].copy()
        row[i] = np.inf   # exclude self
        nearest_k_dists = np.sort(row)[:k]
        scores[i] = np.sum(nearest_k_dists)

    return scores


def single_krum(
    updates: List[List[np.ndarray]],
    f: int
) -> List[np.ndarray]:
    """
    Single-Krum: return the one update with the lowest Krum score.

    Args:
        updates: list of n updates; each update is a list of np.ndarrays (one per layer)
        f:       number of assumed Byzantine clients
    Returns:
        selected: the list of layer params from the best client
    """
    # Flatten each client's update to a single vector for distance computation
    flat = [np.concatenate([p.flatten() for p in u]) for u in updates]
    scores = krum_scores(flat, f)
    best_idx = int(np.argmin(scores))
    return updates[best_idx]


def multi_krum(
    updates: List[List[np.ndarray]],
    f: int,
    m: int = None
) -> List[np.ndarray]:
    """
    Multi-Krum: select top-m clients by lowest Krum score, then average them.

    Args:
        updates: list of n updates (each is list of np.ndarrays per layer)
        f:       number of assumed Byzantine clients
        m:       number of clients to select (default: n - f)
    Returns:
        aggregated: averaged params over selected m clients
    """
    n = len(updates)
    if m is None:
        m = n - f

    flat = [np.concatenate([p.flatten() for p in u]) for u in updates]
    scores = krum_scores(flat, f)
    selected_indices = np.argsort(scores)[:m]

    # Average the selected m updates layer by layer
    aggregated = []
    num_layers = len(updates[0])
    for layer_idx in range(num_layers):
        layer_avg = np.mean(
            [updates[i][layer_idx] for i in selected_indices],
            axis=0
        )
        aggregated.append(layer_avg)

    return aggregated


if __name__ == "__main__":
    # Smoke test: n=10, f=3, 1 malicious update that is far from the rest
    rng = np.random.default_rng(42)
    n, d = 10, 500
    honest_updates  = [rng.normal(0, 0.01, d) for _ in range(7)]
    malicious_updates = [rng.normal(100, 0.01, d) for _ in range(3)]  # far from honest
    all_updates_flat = honest_updates + malicious_updates

    # Wrap as list-of-list (simulate 1-layer model)
    all_updates = [[u] for u in all_updates_flat]

    scores = krum_scores(all_updates_flat, f=3)
    print("Krum scores (lower=more honest):")
    for i, s in enumerate(scores):
        tag = "[MALICIOUS]" if i >= 7 else "[honest]   "
        print(f"  client {i:2d} {tag} score = {s:.2f}")

    selected = single_krum(all_updates, f=3)
    sel_idx = int(np.argmin(scores))
    print(f"\nSingle-Krum selected client {sel_idx} (should be 0–6, i.e. honest)")
    assert sel_idx < 7, "⚠ Krum selected a MALICIOUS client! Check implementation."

    multi = multi_krum(all_updates, f=3, m=7)
    print(f"Multi-Krum aggregated {len(multi)} layers ✓")
    print("✓ Krum smoke test passed — malicious clients excluded")