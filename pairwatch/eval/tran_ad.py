from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pairwatch.eval.metrics import stratified_metrics
from pairwatch.eval.lstm_ad import (
    FEATURE_COLS, SEQ_LEN, normalize_feature_df, to_tensor_per_window,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=BENCH_DIR / "windows_meta.parquet")
    ap.add_argument("--feat", default=BENCH_DIR / "windows_feat.parquet")
    ap.add_argument("--out", default=REPO_ROOT / "runs" / "tranad.json")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-range", action="store_true")
    return ap.parse_args()


class TranADLite(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, layers: int):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.proj_out = nn.Linear(d_model, in_dim)

    def forward(self, x):
        T = x.size(1)
        h = self.proj_in(x) + self.pos[:, :T, :]
        h = self.enc(h)
        return self.proj_out(h)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = pd.read_parquet(args.meta).sort_values("window_id").reset_index(drop=True)
    feat = pd.read_parquet(args.feat)
    print(f"meta {len(meta)} windows, feat {len(feat)} rows")

    X, y = to_tensor_per_window(meta, feat)
    if args.drop_range:
        X = X[:, :, 1:]
        print("(hard mode) dropped range channel")
    print(f"tensor {X.shape}")

    train_mask = (meta["split"] == "train").values
    mu = X[train_mask].mean(axis=(0, 1), keepdims=True)
    sd = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    Xt = torch.from_numpy(Xn).float()

    model = TranADLite(in_dim=Xn.shape[2], d_model=args.d_model,
                       nhead=args.nhead, layers=args.layers)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction="none")

    train_idx = np.where(train_mask)[0]
    train_loader = DataLoader(
        TensorDataset(Xt[train_idx]),
        batch_size=args.batch, shuffle=True,
    )

    model.train()
    for epoch in range(args.epochs):
        total = 0.0; n = 0
        for (xb,) in train_loader:
            recon = model(xb)
            loss = loss_fn(recon, xb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(xb); n += len(xb)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:02d}  train_mse={total / n:.4f}")

    model.eval()
    with torch.no_grad():
        recon = model(Xt)
        per_step = loss_fn(recon, Xt).mean(dim=2)
        scores_mean = per_step.mean(dim=1).cpu().numpy()
        scores_max = per_step.max(dim=1).values.cpu().numpy()

    out = {"model": "tran_ad", "n_train": int(train_mask.sum()),
           "drop_range": bool(args.drop_range), "epochs": args.epochs,
           "d_model": args.d_model, "layers": args.layers}
    for split_name in ("val", "test"):
        mask = (meta["split"] == split_name).values
        if mask.sum() == 0:
            continue
        m_mean = stratified_metrics(y[mask], scores_mean[mask])
        m_max = stratified_metrics(y[mask], scores_max[mask])
        out[split_name] = {"mean_mse": m_mean, "max_mse": m_max}
        a = m_mean["all"]; a_max = m_max["all"]
        print(f"{split_name}: mean-MSE AUROC={a['auroc']:.3f} F1={a['f1']:.3f} | "
              f"max-MSE AUROC={a_max['auroc']:.3f} F1={a_max['f1']:.3f}")
        if "rpo_real_only" in m_mean:
            print(f"   rpo_real_only:  mean AUROC={m_mean['rpo_real_only']['auroc']:.3f} | "
                  f"max AUROC={m_max['rpo_real_only']['auroc']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
