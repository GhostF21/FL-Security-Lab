"""
Coordinate-Wise Trimmed Mean Aggregation (Yin et al., 2018)
Byzantine-resilient aggregation that removes extreme values per coordinate.

Algorithm:
  For each parameter dimension d:
    1. Collect the d-th value from every client update.
    2. Sort the n values.
    3. Trim the lowest beta and highest beta values.
    4. Average the remaining (n - 2*beta) values.

Correctness requirement:
  beta >= f   (beta must be AT LEAST equal to the number of malicious clients f)
  n > 2*beta  (need at least 1 value left after trimming both ends)

  With f malicious clients injecting extreme values:
    - After sorting, malicious values land at the HIGH end (if they inject large vals)
      or LOW end (if small). You must trim AT LEAST f from each end to exclude them all.
    - beta = f is the minimum correct choice.
    - beta = f is also what the original paper uses by default.

Complexity: O(n x d x log n)   (sort each coordinate independently)
"""
from typing import List
import numpy as np


def trimmed_mean(
    updates: List[List[np.ndarray]],
    beta: int,
) -> List[np.ndarray]:
    """
    Coordinate-wise Trimmed Mean aggregation.

    Args:
        updates : List of n client updates.
                  Each update is a list of np.ndarrays (one per model layer).
        beta    : Number of values to trim from EACH end per coordinate.
                  Must satisfy: beta >= f  AND  n > 2*beta.
                  Use beta_from_fraction() to compute this automatically.

    Returns:
        aggregated: List of np.ndarrays (trimmed-mean parameters, one per layer)

    Raises:
        ValueError: If n <= 2*beta (not enough clients left after trimming).
    """
    n = len(updates)
    if n <= 2 * beta:
        raise ValueError(
            f"Not enough clients: n={n} must be > 2*beta={2*beta}. "
            f"Reduce beta or increase num_clients. "
            f"(Current: beta={beta}, need at least {2*beta + 1} clients)"
        )

    num_layers = len(updates[0])
    aggregated = []

    for layer_idx in range(num_layers):
        # Stack all clients' values for this layer: shape (n, *layer_shape)
        stacked = np.stack([updates[i][layer_idx] for i in range(n)], axis=0)

        # Sort along the client axis (axis=0) for each coordinate independently
        sorted_vals = np.sort(stacked, axis=0)

        # Trim beta from each end, average the middle (n - 2*beta) values
        trimmed   = sorted_vals[beta : n - beta]   # shape: (n-2β, *layer_shape)
        layer_mean = np.mean(trimmed, axis=0)       # shape: (*layer_shape)
        aggregated.append(layer_mean)

    return aggregated


def beta_from_fraction(num_clients: int, malicious_fraction: float) -> int:
    """
    Compute the correct beta from the malicious fraction.

    Beta must be >= f (number of malicious clients) to guarantee all malicious
    updates are trimmed from at least one end. Using floor(f/2) is WRONG because
    it only removes half of them.

    Formula: beta = f = ceil(num_clients * malicious_fraction)
    Safety check: ensures n > 2*beta, raises if not satisfiable.

    Args:
        num_clients        : total number of FL clients per round
        malicious_fraction : fraction in [0, 1), e.g. 0.2 for 20%

    Returns:
        beta : int >= 1
    """
    f = int(np.ceil(num_clients * malicious_fraction))  # number of malicious clients

    if f == 0:
        return 0  # no malicious clients — trimming nothing is fine

    beta = f  # must trim AT LEAST f from each end

    if num_clients <= 2 * beta:
        raise ValueError(
            f"Cannot satisfy n > 2*beta: n={num_clients}, f={f}, beta={beta}. "
            f"You need at least {2*beta + 1} clients for {malicious_fraction*100:.0f}% malicious."
        )

    return beta


if __name__ == "__main__":
    # ── Smoke test ───────────────────────────────────────────────────────────
    # Setup: 10 clients, 3 malicious (inject extreme values ~100)
    # Correct beta = f = 3  (trim 3 from each end → 4 honest values remain)
    rng = np.random.default_rng(42)
    n, d = 10, 200
    honest    = [rng.normal(0, 0.01, d) for _ in range(7)]
    malicious = [rng.normal(100, 0.01, d) for _ in range(3)]   # extreme!
    all_updates = [[u] for u in honest + malicious]             # 1-layer model

    # ── Test beta_from_fraction ──────────────────────────────────────────────
    beta = beta_from_fraction(n, malicious_fraction=0.3)
    print(f"β = {beta}  (from 30% malicious, 10 clients)  ← should be 3, not 1")
    assert beta == 3, f"Expected beta=3, got beta={beta}"

    # ── Test trimmed_mean output ─────────────────────────────────────────────
    result   = trimmed_mean(all_updates, beta=beta)
    avg_result = np.mean(result[0])
    print(f"TrimMean result mean: {avg_result:.6f}  (should be ~0, not ~25+)")
    assert abs(avg_result) < 0.5, f"TrimMean FAILED — result={avg_result:.4f}, malicious updates not trimmed!"
    print("✓ Trimmed Mean smoke test passed")

    # ── Verify ValueError on bad beta ────────────────────────────────────────
    try:
        trimmed_mean(all_updates, beta=5)   # 2*5=10, n=10, not > 10
        print("FAIL — should have raised ValueError")
    except ValueError as e:
        print(f"✓ ValueError correctly raised for beta=5, n=10: {e}")

    # ── Show what beta=1 (old wrong formula) produces ─────────────────────────
    wrong_result = trimmed_mean(all_updates, beta=1)
    wrong_mean   = np.mean(wrong_result[0])
    print(f"\n[Debug] With beta=1 (old wrong formula): mean={wrong_mean:.2f}  ← still ~25, malicious survive!")
    print(f"[Debug] With beta=3 (correct):           mean={avg_result:.6f}  ← malicious excluded ✓")