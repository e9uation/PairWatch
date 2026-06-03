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

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"

FEATURE_COLS = ["range_km", "range_rate_km_s",
                "rel_speed_km_s", "plane_angle_deg"]
SEQ_LEN = 336


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=BENCH_DIR / "windows_meta.parquet")
    ap.add_argument("--feat", default=BENCH_DIR / "windows_feat.parquet")
    ap.add_argument("--out", default=REPO_ROOT / "runs" / "lstm_ad.json")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-range", action="store_true",
                    help="Hard-mode ablation: drop range_km from inputs.")
    return ap.parse_args()


def normalize_feature_df(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    out["range_km"] = np.log1p(np.clip(out["range_km"].values, 0, None))
    return out


def to_tensor_per_window(meta: pd.DataFrame, feat: pd.DataFrame
                          ) -> tuple[np.ndarray, np.ndarray]:
    feat = normalize_feature_df(feat)
    arrs, labels = [], []
    for r in meta.itertuples(index=False):
        sub = feat[feat["window_id"] == r.window_id].sort_values("t")
        if len(sub) == 0:
            continue
        x = sub[FEATURE_COLS].to_numpy(dtype=np.float32)
        if len(x) < SEQ_LEN:
            pad = np.repeat(x[-1:], SEQ_LEN - len(x), axis=0)
            x = np.vstack([x, pad])
        else:
            x = x[:SEQ_LEN]
        arrs.append(x); labels.append(int(r.label))
    return np.stack(arrs), np.array(labels)


class LSTMPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, in_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = pd.read_parquet(args.meta)
    feat = pd.read_parquet(args.feat)
    print(f"meta {len(meta)} windows, feat {len(feat)} rows")

    meta = meta.sort_values("window_id").reset_index(drop=True)

    X, y = to_tensor_per_window(meta, feat)
    if args.drop_range:
        X = X[:, :, 1:]
        print("(hard mode) dropped range channel")
    print(f"tensor {X.shape}, labels {y.shape}")

    train_mask = (meta["split"] == "train").values
    mu = X[train_mask].mean(axis=(0, 1), keepdims=True)
    sd = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    Xt = torch.from_numpy(Xn).float()
    inp = Xt[:, :-1, :]; tgt = Xt[:, 1:, :]

    model = LSTMPredictor(in_dim=Xn.shape[2], hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction="none")

    train_idx = np.where(train_mask)[0]
    train_loader = DataLoader(
        TensorDataset(inp[train_idx], tgt[train_idx]),
        batch_size=args.batch, shuffle=True,
    )

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss) * len(xb)
        epoch_loss /= len(train_idx)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:02d}  train_mse={epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        pred_all = model(inp)
        per_step = loss_fn(pred_all, tgt).mean(dim=2)
        scores = per_step.mean(dim=1).cpu().numpy()
        scores_max = per_step.max(dim=1).values.cpu().numpy()

    out = {"model": "lstm_ad", "n_train": int(train_mask.sum()),
           "hidden": args.hidden, "epochs": args.epochs}
    for split_name in ("val", "test"):
        mask = (meta["split"] == split_name).values
        if mask.sum() == 0:
            continue
        m_mean = stratified_metrics(y[mask], scores[mask])
        m_max = stratified_metrics(y[mask], scores_max[mask])
        out[split_name] = {"mean_mse": m_mean, "max_mse": m_max}
        a = m_mean["all"]
        a_max = m_max["all"]
        print(f"{split_name}: mean-MSE AUROC={a['auroc']:.3f} F1={a['f1']:.3f} | "
              f"max-MSE AUROC={a_max['auroc']:.3f} F1={a_max['f1']:.3f}")
        if "rpo_real_only" in m_mean:
            print(f"   rpo_real_only:  mean AUROC={m_mean['rpo_real_only']['auroc']:.3f} | "
                  f"max AUROC={m_max['rpo_real_only']['auroc']:.3f}")

    args_out = Path(args.out)
    args_out.parent.mkdir(parents=True, exist_ok=True)
    args_out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote -> {args_out}")


if __name__ == "__main__":
    main()
