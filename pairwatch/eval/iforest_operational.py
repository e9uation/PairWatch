from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve

from pairwatch.eval.iforest import window_features

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"

OPERATIONAL_NEG_SOURCES = ["clean", "routine_maneuver", "starlink_constellation"]


def f1_at_best(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return float(np.max(f))


def p_at_target_fpr(y, s, target_fpr):
    neg = s[y == 0]; pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - target_fpr)
    pred_pos = (s >= thr).sum()
    tp = ((s >= thr) & (y == 1)).sum()
    return float(tp / max(pred_pos, 1))


def main() -> None:
    meta = pd.read_parquet(BENCH_DIR / "windows_meta.parquet")
    meta["t_start"] = pd.to_datetime(meta["t_start"], utc=True)
    meta["t_end"] = pd.to_datetime(meta["t_end"], utc=True)

    print(f"Building features for {len(meta)} windows...")
    feats = np.stack([window_features(int(r.target_id), r.t_start, r.t_end)
                      for r in meta.itertuples(index=False)])

    train_mask = (meta["split"] == "train").values
    test_mask = (meta["split"] == "test").values

    model = IsolationForest(n_estimators=200, contamination="auto",
                             random_state=0).fit(feats[train_mask])
    scores = -model.score_samples(feats)

    meta_test = meta[test_mask].reset_index(drop=True)
    scores_test = scores[test_mask]
    src = meta_test["source"].apply(lambda x: x.split(":")[0]).values

    real_mask = (src == "rpo_real")
    op_neg_mask = np.isin(src, OPERATIONAL_NEG_SOURCES)
    sub = real_mask | op_neg_mask
    y = real_mask[sub].astype(int)
    s = scores_test[sub]

    auroc = float(roc_auc_score(y, s))
    f1 = f1_at_best(y, s)
    p1 = p_at_target_fpr(y, s, 0.01)

    print(f"\noperational subset (n_pos={y.sum()}, n_neg={(y==0).sum()})")
    print(f"  AUROC    = {auroc:.4f}")
    print(f"  best-F1  = {f1:.4f}")
    print(f"  P@1%FPR  = {p1:.4f}")

    out = {"n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
           "auroc": auroc, "best_f1": f1, "p_at_1pct_fpr": p1}
    (REPO_ROOT / "runs" / "iforest_operational.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
