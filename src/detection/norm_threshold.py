"""
Detector 1 — L2 Norm Threshold (µ+kσ)

Flags clients whose gradient L2 norm exceeds mu + k*sigma
where mu and sigma are computed across all clients per round.

Strengths : Fast O(n), interpretable, near-perfect on gradient scaling
Weaknesses: Blind to label-flip (norms overlap with honest clients)
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def detect_norm_threshold(
    norms: List[float],
    k: float = 2.0,
) -> Tuple[List[int], float]:
    """
    Flag clients whose L2 norm exceeds mu + k*sigma.

    Args:
        norms : List of L2 norms, one per client (for one round).
        k     : Multiplier for sigma.
                - k=2.0 : conservative (default for production — fewer FP)
                - k=1.5 : balanced   (recommended when malicious_fraction >= 20%)
                - k=1.0 : aggressive (highest recall, more FP)

    Returns:
        flags     : List of 0/1 flags, length == len(norms)
        threshold : The computed mu + k*sigma value
    """
    arr       = np.array(norms, dtype=float)
    mu        = arr.mean()
    sigma     = arr.std()
    threshold = mu + k * sigma
    flags     = (arr > threshold).astype(int).tolist()
    return flags, threshold


def evaluate_norm_threshold(
    norm_df: pd.DataFrame,
    k: float = 2.0,
) -> Dict:
    """
    Evaluate norm threshold detector across all rounds.

    Args:
        norm_df : DataFrame with columns: round, client_id, l2_norm, is_malicious
        k       : Sigma multiplier (see detect_norm_threshold for guidance)

    Returns:
        Dict with precision, recall, f1, fpr, and per-round flags DataFrame
    """
    all_true  = []
    all_pred  = []
    all_flags = []

    for rnd, grp in norm_df.groupby("round"):
        norms  = grp["l2_norm"].tolist()
        truth  = grp["is_malicious"].astype(int).tolist()
        flags, thr = detect_norm_threshold(norms, k=k)

        all_true.extend(truth)
        all_pred.extend(flags)
        for (_, row), flag in zip(grp.iterrows(), flags):
            all_flags.append({
                "round"        : rnd,
                "client_id"    : row["client_id"],
                "l2_norm"      : row["l2_norm"],
                "is_malicious" : row["is_malicious"],
                "flagged"      : bool(flag),
                "threshold"    : thr,
            })

    # Compute metrics — handle edge cases where all preds are 0
    prec = precision_score(all_true, all_pred, zero_division=0)
    rec  = recall_score(all_true, all_pred, zero_division=0)
    f1   = f1_score(all_true, all_pred, zero_division=0)

    # False Positive Rate = FP / (FP + TN)
    tn   = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 0)
    fp   = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 1)
    fpr  = fp / (fp + tn + 1e-9)

    return {
        "detector"  : f"NormThreshold(k={k})",
        "precision" : round(prec, 4),
        "recall"    : round(rec,  4),
        "f1"        : round(f1,   4),
        "fpr"       : round(fpr,  4),
        "flags_df"  : pd.DataFrame(all_flags),
    }


if __name__ == "__main__":
    # ── Smoke test ─────────────────────────────────────────────────────────
    # Setup: 10 clients per round, 8 honest (norm ~0.52), 2 malicious (norm ~5.2)
    # k=1.5 is used here because with only 10 clients the malicious norms (~5.2)
    # can fall just below the k=2.0 threshold (~5.09) depending on the random seed.
    # k=1.5 gives threshold ~4.18 — safely below 5.2 — so both malicious clients
    # are always flagged with zero false positives.
    # Production default k=2.0 is appropriate for larger client pools (n >= 20).
    rng = np.random.default_rng(42)
    rows = []
    for r in range(1, 21):
        for c in range(8):
            rows.append({
                "round"        : r,
                "client_id"    : c,
                "l2_norm"      : max(0.1, rng.normal(0.52, 0.06)),
                "is_malicious" : False,
            })
        for c in range(2):
            rows.append({
                "round"        : r,
                "client_id"    : 8 + c,
                "l2_norm"      : max(1.0, rng.normal(5.2, 0.25)),
                "is_malicious" : True,
            })

    df     = pd.DataFrame(rows)
    result = evaluate_norm_threshold(df, k=1.5)   # k=1.5 for n=10 smoke test

    print(f"NormThreshold(k=1.5) — F1: {result['f1']:.4f}  "
          f"P: {result['precision']:.4f}  "
          f"R: {result['recall']:.4f}  "
          f"FPR: {result['fpr']:.4f}")

    assert result["f1"] > 0.9, (
        f"Expected F1 > 0.9 with k=1.5 on gradient scaling data, got {result['f1']:.4f}"
    )
    print("✓ NormThreshold smoke test passed")

    # Also show what k=2.0 gives so the trade-off is visible
    result_k2 = evaluate_norm_threshold(df, k=2.0)
    print(f"\nNormThreshold(k=2.0) — F1: {result_k2['f1']:.4f}  "
          f"P: {result_k2['precision']:.4f}  "
          f"R: {result_k2['recall']:.4f}  "
          f"FPR: {result_k2['fpr']:.4f}")
    print("  (k=2.0 may miss borderline clients at n=10 — use k=1.5 for small pools)")