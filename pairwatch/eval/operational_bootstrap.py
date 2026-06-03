from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve

from pairwatch.eval.lstm_ad import to_tensor_per_window
from pairwatch.eval.significance import train_and_score
from pairwatch.eval.significance_fixA import train_pairwatch_mixed

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"
OPERATIONAL_NEG_SOURCES = ["clean", "routine_maneuver", "starlink_constellation"]
N_BOOT = 1000


def f1_best(y, s):
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return float(np.max(f))


def p_at_1pct_fpr(y, s):
    neg = s[y == 0]; pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 0.99)
    pred_pos = (s >= thr).sum()
    tp = ((s >= thr) & (y == 1)).sum()
    return float(tp / max(pred_pos, 1))


def main() -> None:
    meta = pd.read_parquet(BENCH_DIR / "windows_meta.parquet").sort_values("window_id").reset_index(drop=True)
    feat = pd.read_parquet(BENCH_DIR / "windows_feat.parquet")
    X, y_all = to_tensor_per_window(meta, feat)
    X = X[:, :, 1:]
    train_mask = (meta["split"] == "train").values
    test_mask = (meta["split"] == "test").values
    mu = X[train_mask].mean(axis=(0, 1), keepdims=True)
    sd = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    Xt = torch.from_numpy(Xn).float()
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    seeds = [0, 1, 2, 3, 4]
    methods = ["lstm_ad", "tran_ad", "anomtrans"]
    scores_per_seed = {m: [] for m in methods + ["pairwatch_fixA"]}
    for s in seeds:
        for m in methods:
            print(f"  train {m} seed={s}")
            scores_per_seed[m].append(train_and_score(m, s, Xt, train_idx, 30))
        print(f"  train pairwatch_fixA seed={s}")
        scores_per_seed["pairwatch_fixA"].append(
            train_pairwatch_mixed(s, Xt, train_idx))

    scores_mean = {m: np.mean(np.stack(scores_per_seed[m]), axis=0)[test_idx]
                   for m in scores_per_seed}

    meta_test = meta.iloc[test_idx].reset_index(drop=True)
    src = meta_test["source"].apply(lambda x: x.split(":")[0]).values
    y_test_full = y_all[test_idx]
    real_pos_mask = (y_test_full == 2)
    op_neg_mask = np.isin(src, OPERATIONAL_NEG_SOURCES)

    n_pos = int(real_pos_mask.sum())
    n_neg = int(op_neg_mask.sum())
    print(f"\nOperational subset: n_pos={n_pos} real RPO, n_neg={n_neg} operational negatives")

    rng = np.random.default_rng(0)
    results = {}

    print("\n=== Bootstrap robustness (B=1000): fraction with separation gap > 0 ===")
    print(f"{'method':18s}  {'point gap':>9s}  {'frac gap>0':>10s}  {'frac gap>=0':>11s}")

    for m, s_mean in scores_mean.items():
        s_pos_full = s_mean[real_pos_mask]
        s_neg_full = s_mean[op_neg_mask]
        point_gap = float(s_pos_full.min() - s_neg_full.max())

        n_gap_pos = 0
        n_gap_geq_zero = 0
        aurocs, f1s, p1s = [], [], []
        for _ in range(N_BOOT):
            idx_pos = rng.integers(0, n_pos, size=n_pos)
            idx_neg = rng.integers(0, n_neg, size=n_neg)
            sp = s_pos_full[idx_pos]
            sn = s_neg_full[idx_neg]
            gap = sp.min() - sn.max()
            if gap > 0:
                n_gap_pos += 1
            if gap >= 0:
                n_gap_geq_zero += 1
            s_resampled = np.r_[sp, sn]
            y_resampled = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int)
            try:
                aurocs.append(roc_auc_score(y_resampled, s_resampled))
                f1s.append(f1_best(y_resampled, s_resampled))
                p1s.append(p_at_1pct_fpr(y_resampled, s_resampled))
            except Exception:
                pass

        frac_pos = n_gap_pos / N_BOOT
        frac_geq = n_gap_geq_zero / N_BOOT
        results[m] = {
            "point_gap": point_gap,
            "frac_gap_positive": frac_pos,
            "frac_gap_nonneg": frac_geq,
            "auroc": {
                "point": float(roc_auc_score(
                    np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int),
                    np.r_[s_pos_full, s_neg_full])),
                "ci_lo": float(np.percentile(aurocs, 2.5)),
                "ci_hi": float(np.percentile(aurocs, 97.5)),
            },
            "best_f1": {
                "point": f1_best(
                    np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int),
                    np.r_[s_pos_full, s_neg_full]),
                "ci_lo": float(np.percentile(f1s, 2.5)),
                "ci_hi": float(np.percentile(f1s, 97.5)),
            },
            "p_at_1pct_fpr": {
                "point": p_at_1pct_fpr(
                    np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int),
                    np.r_[s_pos_full, s_neg_full]),
                "ci_lo": float(np.percentile(p1s, 2.5)),
                "ci_hi": float(np.percentile(p1s, 97.5)),
            },
        }
        print(f"{m:18s}  {point_gap:>+9.4f}  {frac_pos:>10.3f}  {frac_geq:>11.3f}")

    print("\n=== Bootstrap 95% CI for operational metrics ===")
    print(f"{'method':18s}  {'AUROC':>20s}  {'best-F1':>20s}  {'P@1%FPR':>20s}")
    for m, r in results.items():
        a = r['auroc']; f1 = r['best_f1']; p1 = r['p_at_1pct_fpr']
        print(f"{m:18s}  "
              f"{a['point']:.3f} [{a['ci_lo']:.3f},{a['ci_hi']:.3f}]  "
              f"{f1['point']:.3f} [{f1['ci_lo']:.3f},{f1['ci_hi']:.3f}]  "
              f"{p1['point']:.3f} [{p1['ci_lo']:.3f},{p1['ci_hi']:.3f}]")

    out_path = REPO_ROOT / "runs" / "operational_bootstrap.json"
    out_path.write_text(json.dumps({"n_pos": n_pos, "n_neg": n_neg, "n_boot": N_BOOT,
                                     "per_method": results}, indent=2))
    print(f"\nWrote -> {out_path}")


if __name__ == "__main__":
    main()
