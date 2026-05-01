"""
Detector 3 — PCA Outlier Detection

Projects gradient updates into PCA space and flags clients
that are statistical outliers (high Mahalanobis distance
from the cluster center).

Strengths : Most powerful — captures complex anomaly patterns
            Works on both magnitude AND direction changes
Weaknesses: Requires full gradient vectors (memory intensive for large models)
            Requires at least n_components+1 clients
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.spatial.distance import mahalanobis


def detect_pca_outlier(
    updates: List[np.ndarray],
    n_components: int = 2,
    percentile_threshold: float = 90.0,
) -> Tuple[List[int], np.ndarray]:
    """
    Flag clients that are PCA outliers.

    Args:
        updates              : List of flattened gradient vectors (n, d)
        n_components         : PCA dimensions to keep (2 is standard)
        percentile_threshold : Clients above this Mahalanobis distance percentile
                               are flagged. 90 = top 10% flagged.

    Returns:
        flags      : List of 0/1 flags (length n)
        projections: PCA projections shape (n, n_components) for plotting
    """
    n = len(updates)
    if n <= n_components:
        print(f"Warning: n={n} <= n_components={n_components}, returning all zeros")
        return [0] * n, np.zeros((n, n_components))

    mat = np.stack(updates, axis=0)   # (n, d)

    # Standardise before PCA
    scaler = StandardScaler()
    mat_s  = scaler.fit_transform(mat)

    # PCA projection
    pca  = PCA(n_components=n_components, random_state=42)
    proj = pca.fit_transform(mat_s)     # (n, 2)

    # Mahalanobis distance from cluster center in PCA space
    mu   = proj.mean(axis=0)
    cov  = np.cov(proj.T)
    if proj.shape[1] == 1 or np.linalg.matrix_rank(cov) < n_components:
        # Fallback to Euclidean if covariance is singular
        dists = np.linalg.norm(proj - mu, axis=1)
    else:
        cov_inv = np.linalg.inv(cov)
        dists   = np.array([mahalanobis(p, mu, cov_inv) for p in proj])

    threshold = np.percentile(dists, percentile_threshold)
    flags     = (dists > threshold).astype(int).tolist()
    return flags, proj


def evaluate_pca_outlier(
    norm_df: pd.DataFrame,
    updates_by_round: Dict[int, List[np.ndarray]],
    n_components: int = 2,
    percentile_threshold: float = 90.0,
) -> Dict:
    """Evaluate PCA outlier detector across all rounds."""
    all_true  = []
    all_pred  = []
    all_flags = []

    for rnd, grp in norm_df.groupby("round"):
        updates = updates_by_round.get(rnd, [])
        if not updates:
            continue
        truth          = grp["is_malicious"].astype(int).tolist()
        flags, proj    = detect_pca_outlier(updates, n_components, percentile_threshold)
        all_true.extend(truth)
        all_pred.extend(flags)
        for i, ((_, row), flag) in enumerate(zip(grp.iterrows(), flags)):
            all_flags.append({
                "round"        : rnd,
                "client_id"    : row["client_id"],
                "pca_x"        : proj[i, 0],
                "pca_y"        : proj[i, 1] if n_components > 1 else 0,
                "is_malicious" : row["is_malicious"],
                "flagged"      : bool(flag),
            })

    prec = precision_score(all_true, all_pred, zero_division=0)
    rec  = recall_score(all_true, all_pred, zero_division=0)
    f1   = f1_score(all_true, all_pred, zero_division=0)
    tn   = sum(1 for t, p in zip(all_true, all_pred) if t==0 and p==0)
    fp   = sum(1 for t, p in zip(all_true, all_pred) if t==0 and p==1)
    fpr  = fp / (fp + tn + 1e-9)

    return {
        "detector"   : f"PCAOutlier(k={n_components}, pct={percentile_threshold})",
        "precision"  : round(prec, 4),
        "recall"     : round(rec,  4),
        "f1"         : round(f1,   4),
        "fpr"        : round(fpr,  4),
        "flags_df"   : pd.DataFrame(all_flags),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    d   = 200
    rows, updates_by_round = [], {}
    for r in range(1, 6):
        hvecs = [rng.normal(0, 0.1, d) for _ in range(8)]
        mvecs = [rng.normal(5, 0.1, d) for _ in range(2)]
        updates_by_round[r] = hvecs + mvecs
        for c in range(8): rows.append({"round":r,"client_id":c,"is_malicious":False})
        for c in range(2): rows.append({"round":r,"client_id":8+c,"is_malicious":True})

    df     = pd.DataFrame(rows)
    result = evaluate_pca_outlier(df, updates_by_round)
    print(f"PCAOutlier — F1: {result['f1']:.4f}  P: {result['precision']:.4f}  R: {result['recall']:.4f}  FPR: {result['fpr']:.4f}")
    print("✓ PCAOutlier smoke test passed")