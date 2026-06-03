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
from pairwatch.eval.significance import (
    train_and_score, bootstrap_auroc, paired_bootstrap_diff, delong_test,
)
from pairwatch.eval.pairwatch import PairWatch, assoc_discrepancy

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"


def train_pairwatch_mixed(seed, Xt, train_idx, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PairWatch(in_dim=Xt.shape[2], d_model=64, nhead=4, layers=2,
                      ablate="fixed_alpha", geom_sim="mixed")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(Xt[train_idx]), batch_size=8, shuffle=True)
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            recon, a, p = model(xb)
            loss = F.mse_loss(recon, xb) + 3.0 * assoc_discrepancy(a, p).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        recon, a, p = model(Xt)
        per_step = F.mse_loss(recon, Xt, reduction="none").mean(dim=2)
        disc = assoc_discrepancy(a, p)
        w = torch.softmax(-disc, dim=-1)
        return (w * per_step).sum(dim=-1).cpu().numpy()


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
    y_test = np.isin(y_all[test_idx], [2, 3]).astype(int)
    y_full = y_all[test_idx]

    methods = ["lstm_ad", "tran_ad", "anomtrans"]
    seeds = [0, 1, 2, 3, 4]
    scores_per_method = {m: [] for m in methods + ["pairwatch_fixA"]}
    for s in seeds:
        for m in methods:
            print(f"train {m} seed={s}")
            scores_per_method[m].append(train_and_score(m, s, Xt, train_idx, 30))
        print(f"train pairwatch_fixA(mixed) seed={s}")
        scores_per_method["pairwatch_fixA"].append(
            train_pairwatch_mixed(s, Xt, train_idx))

    scores_mean = {m: np.mean(np.stack(scores_per_method[m]), axis=0)[test_idx]
                   for m in scores_per_method}

    rng = np.random.default_rng(0)
    result = {"seeds": seeds, "n_boot": 1000}

    print("\n--- bootstrap CI: test all ---")
    for m in scores_mean:
        pt, lo, hi = bootstrap_auroc(y_test, scores_mean[m], 1000, rng)
        print(f"  {m:18s}  AUROC={pt:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")
        result[f"{m}_test_all"] = {"auroc": pt, "ci_lo": lo, "ci_hi": hi}

    sub_mask = np.isin(y_full, [0, 1]) | (y_full == 2)
    y_sub = (y_full[sub_mask] == 2).astype(int)
    print(f"\n--- bootstrap CI: rpo_real (n={sub_mask.sum()}, pos={y_sub.sum()}) ---")
    for m in scores_mean:
        pt, lo, hi = bootstrap_auroc(y_sub, scores_mean[m][sub_mask], 1000, rng)
        print(f"  {m:18s}  AUROC={pt:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")
        result[f"{m}_rpo_real"] = {"auroc": pt, "ci_lo": lo, "ci_hi": hi}

    pw = scores_mean["pairwatch_fixA"]
    print("\n--- paired bootstrap & DeLong: PairWatch_FixA vs baseline (test all) ---")
    for m in methods:
        d = paired_bootstrap_diff(y_test, pw, scores_mean[m], 1000, rng)
        dl = delong_test(y_test, pw, scores_mean[m])
        d.update(dl)
        print(f"  vs {m:12s}: diff={d['diff_mean']:+.4f} "
              f"95%CI=[{d['diff_ci_lo']:+.4f}, {d['diff_ci_hi']:+.4f}]  "
              f"DeLong p={d.get('delong_p', float('nan')):.4g}")
        result[f"vs_{m}_test_all"] = d

    pw_sub = pw[sub_mask]
    print("\n--- paired bootstrap & DeLong: PairWatch_FixA vs baseline (rpo_real) ---")
    for m in methods:
        d = paired_bootstrap_diff(y_sub, pw_sub, scores_mean[m][sub_mask], 1000, rng)
        dl = delong_test(y_sub, pw_sub, scores_mean[m][sub_mask])
        d.update(dl)
        print(f"  vs {m:12s}: diff={d['diff_mean']:+.4f} "
              f"95%CI=[{d['diff_ci_lo']:+.4f}, {d['diff_ci_hi']:+.4f}]  "
              f"DeLong p={d.get('delong_p', float('nan')):.4g}")
        result[f"vs_{m}_rpo_real"] = d

    (REPO_ROOT / "runs" / "significance_fixA.json").write_text(
        json.dumps(result, indent=2, default=str))
    print(f"\nWrote runs/significance_fixA.json")


if __name__ == "__main__":
    main()
