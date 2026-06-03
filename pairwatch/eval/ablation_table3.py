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
from pairwatch.eval.pairwatch import PairWatch, assoc_discrepancy

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"

CHANNELS = ["range_rate", "rel_speed", "plane_angle"]


def train_pairwatch_variant(seed, Xt, train_idx,
                            ablate="fixed_alpha", geom_sim="mixed",
                            epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PairWatch(in_dim=Xt.shape[2], d_model=64, nhead=4, layers=2,
                      ablate=ablate, geom_sim=geom_sim)
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


def evaluate(scores, y_all, test_idx, y_test, sub_mask, y_sub):
    s_test = scores[test_idx]
    auroc_all = float(roc_auc_score(y_test, s_test))
    auroc_real = float(roc_auc_score(y_sub, s_test[sub_mask]))
    return auroc_all, auroc_real


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
    sub_mask = np.isin(y_full, [0, 1]) | (y_full == 2)
    y_sub = (y_full[sub_mask] == 2).astype(int)

    seeds = [0, 1, 2, 3, 4]

    variants = [
        ("full_mixed",   dict(ablate="fixed_alpha",   geom_sim="mixed"),  None),
        ("no_geom",      dict(ablate="temporal_only", geom_sim="mixed"),  None),
        ("sim_raw",      dict(ablate="fixed_alpha",   geom_sim="raw"),    None),
        ("sim_abs",      dict(ablate="fixed_alpha",   geom_sim="abs"),    None),
    ]
    for ch_idx, ch_name in enumerate(CHANNELS):
        variants.append(
            (f"no_{ch_name}",
             dict(ablate="fixed_alpha", geom_sim="mixed"),
             ch_idx))

    results = {}
    for var_name, kwargs, zero_ch in variants:
        print(f"\n=== {var_name}  kwargs={kwargs}  zero_ch={zero_ch} ===")
        aurocs_all, aurocs_real = [], []
        for s in seeds:
            if zero_ch is not None:
                Xz = Xn.copy()
                Xz[:, :, zero_ch] = 0.0
                Xt_eff = torch.from_numpy(Xz).float()
            else:
                Xt_eff = Xt
            scores = train_pairwatch_variant(s, Xt_eff, train_idx, **kwargs)
            a_all, a_real = evaluate(scores, y_all, test_idx, y_test,
                                     sub_mask, y_sub)
            aurocs_all.append(a_all); aurocs_real.append(a_real)
            print(f"  seed {s}: all={a_all:.4f}  rpo_real={a_real:.4f}")
        results[var_name] = {
            "seeds": seeds,
            "all": aurocs_all,
            "rpo_real": aurocs_real,
            "mean_all": float(np.mean(aurocs_all)),
            "std_all": float(np.std(aurocs_all, ddof=0)),
            "mean_real": float(np.mean(aurocs_real)),
            "std_real": float(np.std(aurocs_real, ddof=0)),
        }

    print("\n\n=== SUMMARY (across-seed mean ± std, 5 seeds, current benchmark) ===")
    print(f"{'variant':25s}  {'test all':>15s}  {'test rpo_real':>15s}")
    for k, v in results.items():
        print(f"{k:25s}  {v['mean_all']:.3f}±{v['std_all']:.3f}    "
              f"{v['mean_real']:.3f}±{v['std_real']:.3f}")

    out_path = REPO_ROOT / "runs" / "ablation_table3.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote -> {out_path}")


if __name__ == "__main__":
    main()
