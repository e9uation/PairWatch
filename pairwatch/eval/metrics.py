from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve,
)


def to_binary(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, [2, 3]).astype(int)


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    y = to_binary(labels)
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "n_pos": int(y.sum()),
                "auroc": float("nan"), "ap": float("nan"),
                "f1": float("nan"), "p_at_1pct_fpr": float("nan"),
                "threshold_at_1pct_fpr": float("nan")}

    auroc = roc_auc_score(y, scores)
    ap = average_precision_score(y, scores)

    p, r, thresh = precision_recall_curve(y, scores)
    f1s = 2 * p * r / np.clip(p + r, 1e-9, None)
    best = int(np.nanargmax(f1s))
    f1 = float(f1s[best])

    fpr, tpr, roc_thresh = roc_curve(y, scores)
    idx = np.searchsorted(fpr, 0.01, side="right") - 1
    if idx < 0:
        p_at = float("nan"); thr = float("nan")
    else:
        thr = float(roc_thresh[idx])
        pred = (scores >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        p_at = float(tp / max(tp + fp, 1))

    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": float(auroc),
        "ap": float(ap),
        "f1": f1,
        "p_at_1pct_fpr": p_at,
        "threshold_at_1pct_fpr": thr,
    }


def stratified_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    out = {"all": compute_metrics(labels, scores)}
    neg_mask = np.isin(labels, [0, 1])
    for pos_label, name in [(2, "rpo_real_only"), (3, "rpo_synth_only")]:
        mask = neg_mask | (labels == pos_label)
        if (labels[mask] == pos_label).sum() < 2:
            continue
        out[name] = compute_metrics(labels[mask], scores[mask])
    return out
