from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score

from pairwatch.eval.lstm_ad import to_tensor_per_window
from pairwatch.eval.significance import train_and_score
from pairwatch.eval.significance_fixA import train_pairwatch_mixed

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"

REGIME_LABELS = {
    44797: "Cosmos 2542 (97deg SSO)",
    42798: "Cosmos 2519 (98deg SSO)",
    53323: "Cosmos 2558 (97deg SSO)",
    44937: "Starlink-1079 (53deg)",
}

OPERATIONAL_NEG_SOURCES = ["clean", "routine_maneuver", "starlink_constellation"]


def f1_at_best(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return float(np.max(f)), float(t[int(np.argmax(f[:-1]))]) if len(t) else float("nan")


def p_at_target_fpr(y, s, target_fpr):
    neg = s[y == 0]; pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    thr = np.quantile(neg, 1 - target_fpr)
    pred_pos = (s >= thr).sum()
    tp = ((s >= thr) & (y == 1)).sum()
    prec = tp / max(pred_pos, 1)
    return float(prec), float(thr)


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
    tgt = meta_test["target_id"].values
    y_test_full = y_all[test_idx]
    y_test_bin = np.isin(y_test_full, [2, 3]).astype(int)
    y_real = (y_test_full == 2).astype(int)
    y_synth = (y_test_full == 3).astype(int)

    operational_neg_mask = np.isin(src, OPERATIONAL_NEG_SOURCES)
    real_pos_mask = (y_test_full == 2)
    formation_neg_mask = (src == "formation_flying")

    result = {"seeds": seeds, "n_test": int(test_mask.sum()),
              "n_real_rpo": int(real_pos_mask.sum()),
              "n_operational_neg": int(operational_neg_mask.sum()),
              "n_formation_neg": int(formation_neg_mask.sum())}

    print("\n=== per-regime AUROC: real RPO of target X vs operational negs ===")
    per_regime = {}
    for m in scores_mean:
        per_regime[m] = {}
        s = scores_mean[m]
        for t, name in REGIME_LABELS.items():
            pos = real_pos_mask & (tgt == t)
            mask = pos | operational_neg_mask
            n_pos = int(pos.sum())
            if n_pos == 0:
                per_regime[m][name] = {"auroc": None, "n_pos": 0}
                continue
            y = pos[mask].astype(int)
            auroc = float(roc_auc_score(y, s[mask]))
            per_regime[m][name] = {"auroc": auroc, "n_pos": n_pos,
                                    "n_neg": int(operational_neg_mask.sum())}
        per_regime[m]["_synth_per_regime"] = {}
        for t, name in REGIME_LABELS.items():
            pos = (y_test_full == 3) & (tgt == t)
            mask = pos | operational_neg_mask
            n_pos = int(pos.sum())
            if n_pos == 0:
                per_regime[m]["_synth_per_regime"][name] = {"auroc": None, "n_pos": 0}
                continue
            y = pos[mask].astype(int)
            auroc = float(roc_auc_score(y, s[mask]))
            per_regime[m]["_synth_per_regime"][name] = {
                "auroc": auroc, "n_pos": n_pos,
                "n_neg": int(operational_neg_mask.sum())}
    result["per_regime"] = per_regime

    methods_disp = list(scores_mean.keys())
    print(f"{'regime':30s}  " + "  ".join(f"{m:>14s}" for m in methods_disp))
    for t, name in REGIME_LABELS.items():
        cells = []
        for m in methods_disp:
            a = per_regime[m][name]["auroc"]
            cells.append(f"{a:>14.3f}" if a is not None else f"{'n/a':>14s}")
        print(f"{name:30s}  " + "  ".join(cells))

    print("\n=== operational separability (real RPO vs operational negs) ===")
    print(f"  n_real={real_pos_mask.sum()}  n_op_neg={operational_neg_mask.sum()}")
    op_sep = {}
    for m, s in scores_mean.items():
        s_pos = s[real_pos_mask]
        s_neg = s[operational_neg_mask]
        min_pos = float(s_pos.min())
        max_neg = float(s_neg.max())
        gap = min_pos - max_neg
        perfect = bool(gap > 0)
        n_overlap_neg = int((s_neg >= min_pos).sum())
        n_below_max_neg_pos = int((s_pos <= max_neg).sum())
        op_sep[m] = {
            "min_pos": min_pos, "max_neg": max_neg, "gap": gap,
            "perfect_separation": perfect,
            "n_neg_above_min_pos": n_overlap_neg,
            "n_pos_below_max_neg": n_below_max_neg_pos,
        }
        flag = "PERFECT" if perfect else f"overlap (gap={gap:+.4f})"
        print(f"  {m:18s}  min_pos={min_pos:.4f}  max_neg={max_neg:.4f}  {flag}")
    result["operational_separability"] = op_sep

    print("\n=== operational metrics (P@1%FPR, best-F1) ===")
    sub_mask = np.isin(y_test_full, [0, 1]) | (y_test_full == 2)
    y_sub = (y_test_full[sub_mask] == 2).astype(int)
    op_sub_mask = real_pos_mask | operational_neg_mask
    y_op = real_pos_mask[op_sub_mask].astype(int)
    op_metrics = {}
    print(f"{'method':18s} {'mode':16s}  {'AUROC':>7s}  {'P@1%FPR':>8s}  {'best-F1':>8s}")
    for m, s in scores_mean.items():
        op_metrics[m] = {}
        for mode, y, s_eff in [
            ("test_all", y_test_bin, s),
            ("test_rpo_real", y_sub, s[sub_mask]),
            ("test_operational", y_op, s[op_sub_mask]),
        ]:
            auroc = float(roc_auc_score(y, s_eff))
            p1, _ = p_at_target_fpr(y, s_eff, 0.01)
            f1, _ = f1_at_best(y, s_eff)
            op_metrics[m][mode] = {"auroc": auroc, "p_at_1pct_fpr": p1, "best_f1": f1}
            print(f"{m:18s} {mode:16s}  {auroc:>7.3f}  {p1:>8.3f}  {f1:>8.3f}")
    result["operational_metrics"] = op_metrics

    from pairwatch.eval.significance import delong_test
    print("\n=== DeLong (operational subset): PairWatch vs baseline ===")
    pw_op = scores_mean["pairwatch_fixA"][op_sub_mask]
    delong_op = {}
    for m in ["lstm_ad", "tran_ad", "anomtrans"]:
        s_op = scores_mean[m][op_sub_mask]
        try:
            d = delong_test(y_op, pw_op, s_op)
            delong_op[f"vs_{m}"] = d
            print(f"  vs {m:12s}  AUROC_A={d.get('auroc_A', float('nan')):.3f}  "
                  f"AUROC_B={d.get('auroc_B', float('nan')):.3f}  "
                  f"DeLong p={d.get('delong_p', float('nan')):.4g}  "
                  f"z={d.get('delong_z', float('nan')):.3f}")
        except Exception as ex:
            print(f"  vs {m:12s}  DeLong failed: {ex}")
            delong_op[f"vs_{m}"] = {"error": str(ex)}
    result["delong_operational"] = delong_op

    out_path = REPO_ROOT / "runs" / "operational_and_regime.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote -> {out_path}")


if __name__ == "__main__":
    main()
