from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pairwatch.eval.lstm_ad import to_tensor_per_window, LSTMPredictor
from pairwatch.eval.tran_ad import TranADLite
from pairwatch.eval.anomaly_transformer import (
    AnomalyTransformer, assoc_discrepancy as at_disc,
)
from pairwatch.eval.pairwatch import PairWatch, assoc_discrepancy as pw_disc

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_epoch(model, loader, opt, kind: str) -> float:
    model.train()
    t0 = time.perf_counter()
    for (xb,) in loader:
        if kind == "lstm_ad":
            ib = xb[:, :-1, :]; tb = xb[:, 1:, :]
            pred = model(ib)
            loss = F.mse_loss(pred, tb)
        elif kind == "tran_ad":
            recon = model(xb)
            loss = F.mse_loss(recon, xb)
        else:
            recon, a, p = model(xb)
            disc_fn = at_disc if kind == "anomtrans" else pw_disc
            loss = F.mse_loss(recon, xb) + 3.0 * disc_fn(a, p).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return time.perf_counter() - t0


def time_inference(model, x_one: torch.Tensor, kind: str, n_iter: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = forward(model, x_one, kind)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = forward(model, x_one, kind)
    return (time.perf_counter() - t0) / n_iter


def forward(model, x, kind):
    if kind == "lstm_ad":
        return model(x[:, :-1, :])
    if kind == "tran_ad":
        return model(x)
    return model(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-warmup", type=int, default=5)
    ap.add_argument("--epochs-measure", type=int, default=10)
    ap.add_argument("--n-infer", type=int, default=100)
    ap.add_argument("--out", default=REPO_ROOT / "runs" / "compute_cost.json")
    args = ap.parse_args()

    meta = pd.read_parquet(BENCH_DIR / "windows_meta.parquet").sort_values("window_id").reset_index(drop=True)
    feat = pd.read_parquet(BENCH_DIR / "windows_feat.parquet")
    X, _ = to_tensor_per_window(meta, feat)
    X = X[:, :, 1:]
    train_mask = (meta["split"] == "train").values
    mu = X[train_mask].mean(axis=(0, 1), keepdims=True)
    sd = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    Xt = torch.from_numpy(Xn).float()
    train_idx = np.where(train_mask)[0]
    in_dim = Xn.shape[2]
    loader = DataLoader(TensorDataset(Xt[train_idx]), batch_size=8, shuffle=True)
    x_one = Xt[:1]

    methods = {
        "lstm_ad":   lambda: LSTMPredictor(in_dim=in_dim, hidden=64),
        "tran_ad":   lambda: TranADLite(in_dim=in_dim, d_model=64, nhead=4, layers=2),
        "anomtrans": lambda: AnomalyTransformer(in_dim=in_dim, d_model=64, nhead=4, layers=2),
        "pairwatch": lambda: PairWatch(in_dim=in_dim, d_model=64, nhead=4, layers=2,
                                        ablate="fixed_alpha", geom_sim="mixed"),
    }

    out = {}
    for name, ctor in methods.items():
        torch.manual_seed(0); np.random.seed(0)
        model = ctor()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        params = n_params(model)
        for _ in range(args.epochs_warmup):
            time_epoch(model, loader, opt, name)
        times_train = [time_epoch(model, loader, opt, name)
                       for _ in range(args.epochs_measure)]
        t_infer = time_inference(model, x_one, name, args.n_infer)
        out[name] = {
            "params": int(params),
            "epoch_s_mean": float(np.mean(times_train)),
            "epoch_s_std": float(np.std(times_train)),
            "inference_ms_per_window": float(t_infer * 1000),
        }
        print(f"{name:12s}  params={params:>9,}  "
              f"epoch={np.mean(times_train):.3f}±{np.std(times_train):.3f}s  "
              f"infer={t_infer*1000:.2f} ms/window")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote -> {args.out}")

    pw = out["pairwatch"]
    print("\n--- relative to PairWatch ---")
    for name, m in out.items():
        print(f"  {name:12s}  params ×{m['params']/pw['params']:.2f}  "
              f"epoch ×{m['epoch_s_mean']/pw['epoch_s_mean']:.2f}  "
              f"infer ×{m['inference_ms_per_window']/pw['inference_ms_per_window']:.2f}")


if __name__ == "__main__":
    main()
