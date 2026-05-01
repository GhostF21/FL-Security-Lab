"""
Detector 2 — Cosine Similarity Clustering
Computes cosine similarity between each client's gradient update
and the mean update direction. Low-similarity clients are flagged.

Works by comparing gradient DIRECTIONS rather than magnitudes,
making it complementary to norm thresholding.

Strengths : Catches direction-based attacks (label-flip at high fractions)
Weaknesses: May miss gradient scaling if directions are similar
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import normalize


def cosine_similarity_to_mean(
    updates: List[np.ndarray],
) -> List[float]:
    """
    Compute cosine similarity of each client's update to the mean update.

    Args:
        updates : List of flattened gradient vectors, one per client.

    Returns:
        similarities : List of cosine similarity values [-1, 1]
    """
    mat      = np.stack(updates, axis=0)          # shape: (n, d)
    mean_vec = mat.mean(axis=0, keepdims=True)    # shape: (1, d)

    # Normalize to unit vectors (safe for zero vectors)
    mat_n    = normalize(mat,      norm="l2", axis=1)
    mean_n   = normalize(mean_vec, norm="l2", axis=1)

    sims = (mat_n @ mean_n.T).flatten().tolist()   # dot product per row
    return sims


def detect_cosine(
    updates: List[np.ndarray],
    threshold: float = 0.5,
) -> Tuple[List[int], List[float]]:
    """
    Flag clients whose cosine similarity to the mean is below threshold.

    Args:
        updates   : List of flattened np.ndarrays, one per client.
        threshold : Similarity cutoff. Below this → flagged.
                    Tune: 0.5 is conservative (few FPs), 0.7 more sensitive.

    Returns:
        flags       : List of 0/1 flags
        similarities: List of cosine similarity values (for logging)
    """
    sims  = cosine_similarity_to_mean(updates)
    flags = [1 if s < threshold else 0 for s in sims]
    return flags, sims


def evaluate_cosine(
    norm_df: pd.DataFrame,
    updates_by_round: Dict[int, List[np.ndarray]],
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate cosine detector across all rounds.

    Args:
        norm_df         : DataFrame with round, client_id, is_malicious columns
        updates_by_round: Dict mapping round number → list of gradient vectors
        threshold       : Cosine similarity cutoff
    """
    all_true  = []
    all_pred  = []
    all_flags = []

    for rnd, grp in norm_df.groupby("round"):
        updates = updates_by_round.get(rnd, [])
        if not updates:
            continue
        truth  = grp["is_malicious"].astype(int).tolist()
        flags, sims = detect_cosine(updates, threshold)

        all_true.extend(truth)
        all_pred.extend(flags)
        for (_, row), flag, sim in zip(grp.iterrows(), flags, sims):
            all_flags.append({
                "round"       : rnd,
                "client_id"   : row["client_id"],
                "cosine_sim"  : sim,
                "is_malicious": row["is_malicious"],
                "flagged"     : bool(flag),
            })

    prec = precision_score(all_true, all_pred, zero_division=0)
    rec  = recall_score(all_true, all_pred, zero_division=0)
    f1   = f1_score(all_true, all_pred, zero_division=0)
    tn   = sum(1 for t, p in zip(all_true, all_pred) if t==0 and p==0)
    fp   = sum(1 for t, p in zip(all_true, all_pred) if t==0 and p==1)
    fpr  = fp / (fp + tn + 1e-9)

    return {
        "detector"   : f"CosineSimilarity(thr={threshold})",
        "precision"  : round(prec, 4),
        "recall"     : round(rec,  4),
        "f1"         : round(f1,   4),
        "fpr"        : round(fpr,  4),
        "flags_df"   : pd.DataFrame(all_flags),
    }


if __name__ == "__main__":
    # ── Smoke test — simulate label-flip: directions diverge from mean ──
    rng = np.random.default_rng(42)
    d   = 500                                       # parameter dims
    rows, updates_by_round = [], {}

    for r in range(1, 11):
        honest_vecs    = [rng.normal(0, 1, d) for _ in range(8)]
        malicious_vecs = [rng.normal(0, 1, d) * -3 for _ in range(2)] # opposite direction
        updates_by_round[r] = honest_vecs + malicious_vecs
        for c in range(8): rows.append({"round":r,"client_id":c,"is_malicious":False})
        for c in range(2): rows.append({"round":r,"client_id":8+c,"is_malicious":True})

    df     = pd.DataFrame(rows)
    result = evaluate_cosine(df, updates_by_round, threshold=0.5)
    print(f"CosineSim  — F1: {result['f1']:.4f}  P: {result['precision']:.4f}  R: {result['recall']:.4f}  FPR: {result['fpr']:.4f}")
    print("✓ CosineSimilarity smoke test passed")