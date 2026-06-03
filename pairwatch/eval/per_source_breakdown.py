from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from pairwatch.eval.lstm_ad import to_tensor_per_window
from pairwatch.eval.significance import train_and_score
from pairwatch.eval.significance_fixA import train_pairwatch_mixed

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"


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
    methods = ["lstm_ad", "tran_ad", "anomtrans", "pairwatch_fixA"]
    scores_per = {m: [] for m in methods}
    for s in seeds:
        for m in ["lstm_ad", "tran_ad", "anomtrans"]:
            print(f"  train {m} seed={s}")
            scores_per[m].append(train_and_score(m, s, Xt, train_idx, 30))
        print(f"  train pairwatch_fixA seed={s}")
        scores_per["pairwatch_fixA"].append(
            train_pairwatch_mixed(s, Xt, train_idx))
    scores_mean = {m: np.mean(np.stack(scores_per[m]), axis=0)[test_idx]
                   for m in methods}

    meta_test = meta.iloc[test_idx].reset_index(drop=True)
    meta_test["src"] = meta_test["source"].apply(lambda x: x.split(":")[0])
    print("\n--- test split by source ---")
    print(meta_test.groupby(["src", "label"]).size().unstack(fill_value=0))

    pos_sources = ["rpo_real", "rpo_synth"]
    neg_sources = ["clean", "routine_maneuver", "starlink_constellation",
                   "formation_flying"]
    src_arr = meta_test["src"].values

    print("\n--- per-(positive,negative) source AUROC (5-seed mean scores) ---")
    print(f"{'pos':>15s} vs {'neg':>22s}  " + "  ".join(f"{m:>14s}" for m in methods))
    breakdown = {}
    for pos in pos_sources:
        for neg in neg_sources:
            mask = (src_arr == pos) | (src_arr == neg)
            y = (src_arr[mask] == pos).astype(int)
            n_pos, n_neg = int(y.sum()), int((y == 0).sum())
            if n_pos < 2 or n_neg < 2:
                continue
            row = {"pos": pos, "neg": neg, "n_pos": n_pos, "n_neg": n_neg}
            vals = {}
            for m in methods:
                try:
                    auroc = float(roc_auc_score(y, scores_mean[m][mask]))
                except ValueError:
                    auroc = float("nan")
                vals[m] = auroc
            row.update(vals)
            breakdown[f"{pos}__vs__{neg}"] = row
            print(f"{pos:>15s} vs {neg:>22s}  "
                  + "  ".join(f"{vals[m]:>14.3f}" for m in methods))

    print("\n--- rpo_real vs ALL negatives (security metric) ---")
    mask = (src_arr == "rpo_real") | np.isin(src_arr, neg_sources)
    y = (src_arr[mask] == "rpo_real").astype(int)
    sec_vals = {}
    for m in methods:
        try:
            auroc = float(roc_auc_score(y, scores_mean[m][mask]))
        except ValueError:
            auroc = float("nan")
        sec_vals[m] = auroc
        print(f"  {m:14s}  AUROC={auroc:.3f}")
    breakdown["rpo_real__vs__ALL_negatives"] = {
        **{"n_pos": int(y.sum()), "n_neg": int((y == 0).sum())},
        **sec_vals,
    }

    (REPO_ROOT / "runs" / "per_source.json").write_text(
        json.dumps(breakdown, indent=2, default=str))
    print(f"\nWrote runs/per_source.json")


if __name__ == "__main__":
    main()
