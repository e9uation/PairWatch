from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pairwatch.eval.metrics import stratified_metrics
from pairwatch.eval.lstm_ad import SEQ_LEN, to_tensor_per_window

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=BENCH_DIR / "windows_meta.parquet")
    ap.add_argument("--feat", default=BENCH_DIR / "windows_feat.parquet")
    ap.add_argument("--out", default=REPO_ROOT / "runs" / "pairwatch.json")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-assoc", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop-range", action="store_true")
    ap.add_argument("--ablate", default="full",
                    choices=["full", "geom_only", "fixed_alpha", "temporal_only"],
                    help="ablation variant: full | geom_only | fixed_alpha | temporal_only")
    ap.add_argument("--seq-len", type=int, default=336,
                    help="effective sequence length (truncate from window start)")
    ap.add_argument("--geom-sim", default="raw",
                    choices=["raw", "abs", "squared", "mixed"],
                    help="Fix A: geometric similarity variant.")
    return ap.parse_args()


class PairAwareAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, geom_dim: int,
                 ablate: str = "full", geom_sim: str = "raw"):
        super().__init__()
        self.h = nhead
        self.d = d_model // nhead
        self.ablate = ablate
        self.geom_sim = geom_sim
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.sigma_proj = nn.Linear(d_model, nhead)
        in_geom_dim = geom_dim * (2 if geom_sim == "mixed" else 1)
        self.beta_proj = nn.Linear(in_geom_dim, nhead)
        self.alpha = nn.Parameter(torch.zeros(nhead))
        self.out = nn.Linear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.d)

    def forward(self, x: torch.Tensor, geom: torch.Tensor):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)

        sigma = F.softplus(self.sigma_proj(x)).clamp_min(0.5).permute(0, 2, 1)
        idx = torch.arange(T, device=x.device).float()
        dt2 = (idx.unsqueeze(0) - idx.unsqueeze(1)).pow(2)  # (T,T)
        log_t = -0.5 * dt2.unsqueeze(0).unsqueeze(0) / (sigma.unsqueeze(-1).pow(2))

        if self.geom_sim == "abs":
            geom_eff = geom.abs()
        elif self.geom_sim == "squared":
            geom_eff = geom.pow(2)
        elif self.geom_sim == "mixed":
            geom_eff = torch.cat([geom, geom.abs()], dim=-1)
        else:
            geom_eff = geom
        beta = F.softplus(self.beta_proj(geom_eff)).clamp_min(0.5).permute(0, 2, 1)
        diff = geom_eff.unsqueeze(2) - geom_eff.unsqueeze(1)
        dg2 = diff.pow(2).sum(-1)
        log_g = -0.5 * dg2.unsqueeze(1) / (beta.unsqueeze(-1).pow(2))

        if self.ablate == "fixed_alpha":
            alpha = torch.ones_like(self.alpha).view(1, -1, 1, 1)
        else:
            alpha = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        if self.ablate == "geom_only":
            prior = torch.softmax(alpha * log_g, dim=-1)
        elif self.ablate == "temporal_only":
            prior = torch.softmax(log_t, dim=-1)
        else:
            prior = torch.softmax(log_t + alpha * log_g, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out(out), attn, prior


class PairAwareLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, geom_dim: int,
                 ablate: str = "full", geom_sim: str = "raw"):
        super().__init__()
        self.attn = PairAwareAttention(d_model, nhead, geom_dim,
                                        ablate=ablate, geom_sim=geom_sim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x, geom):
        h, attn, prior = self.attn(x, geom)
        x = self.n1(x + h)
        x = self.n2(x + self.ff(x))
        return x, attn, prior


class PairWatch(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, layers: int,
                 ablate: str = "full", geom_sim: str = "raw"):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            PairAwareLayer(d_model, nhead, geom_dim=in_dim,
                            ablate=ablate, geom_sim=geom_sim)
            for _ in range(layers)
        ])
        self.out_proj = nn.Linear(d_model, in_dim)

    def forward(self, x):
        T = x.size(1)
        h = self.in_proj(x) + self.pos[:, :T, :]
        attns, priors = [], []
        for blk in self.blocks:
            h, a, p = blk(h, x)
            attns.append(a); priors.append(p)
        recon = self.out_proj(h)
        return recon, attns, priors


def assoc_discrepancy(attns, priors) -> torch.Tensor:
    eps = 1e-9
    discreps = []
    for a, p in zip(attns, priors):
        kl1 = (p * (torch.log(p + eps) - torch.log(a + eps))).sum(-1)
        kl2 = (a * (torch.log(a + eps) - torch.log(p + eps))).sum(-1)
        discreps.append(0.5 * (kl1 + kl2).mean(dim=1))
    return torch.stack(discreps, dim=0).mean(dim=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    meta = pd.read_parquet(args.meta).sort_values("window_id").reset_index(drop=True)
    feat = pd.read_parquet(args.feat)
    print(f"meta {len(meta)} windows, feat {len(feat)} rows")

    X, y = to_tensor_per_window(meta, feat)
    if args.drop_range:
        X = X[:, :, 1:]; print("(hard mode) dropped range channel")
    if args.seq_len < X.shape[1]:
        X = X[:, :args.seq_len, :]
        print(f"(seq sweep) truncated to seq_len={args.seq_len}")
    print(f"tensor {X.shape}")

    train_mask = (meta["split"] == "train").values
    mu = X[train_mask].mean(axis=(0, 1), keepdims=True)
    sd = X[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)
    Xt = torch.from_numpy(Xn).float()

    model = PairWatch(in_dim=Xn.shape[2], d_model=args.d_model,
                       nhead=args.nhead, layers=args.layers,
                       ablate=args.ablate, geom_sim=args.geom_sim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_idx = np.where(train_mask)[0]
    train_loader = DataLoader(
        TensorDataset(Xt[train_idx]),
        batch_size=args.batch, shuffle=True,
    )

    model.train()
    for epoch in range(args.epochs):
        total = 0.0; n = 0
        for (xb,) in train_loader:
            recon, attns, priors = model(xb)
            mse = F.mse_loss(recon, xb)
            disc = assoc_discrepancy(attns, priors).mean()
            loss = mse + args.lambda_assoc * disc
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach()) * len(xb); n += len(xb)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:02d}  loss={total / n:.4f}")

    model.eval()
    with torch.no_grad():
        recon, attns, priors = model(Xt)
        per_step_mse = F.mse_loss(recon, Xt, reduction="none").mean(dim=2)
        disc = assoc_discrepancy(attns, priors)
        weights = torch.softmax(-disc, dim=-1)
        score_at = (weights * per_step_mse).sum(dim=-1).cpu().numpy()
        score_disc = disc.mean(dim=-1).cpu().numpy()
        score_mse = per_step_mse.mean(dim=-1).cpu().numpy()

    out = {"model": "pairwatch", "n_train": int(train_mask.sum()),
           "drop_range": bool(args.drop_range), "epochs": args.epochs,
           "d_model": args.d_model, "layers": args.layers,
           "lambda_assoc": args.lambda_assoc}
    for split_name in ("val", "test"):
        mask = (meta["split"] == split_name).values
        if mask.sum() == 0:
            continue
        m_at = stratified_metrics(y[mask], score_at[mask])
        m_disc = stratified_metrics(y[mask], score_disc[mask])
        m_mse = stratified_metrics(y[mask], score_mse[mask])
        out[split_name] = {"at_score": m_at, "discrepancy": m_disc, "mse": m_mse}
        a = m_at["all"]
        print(f"{split_name}: AT-score AUROC={a['auroc']:.3f} F1={a['f1']:.3f} | "
              f"disc AUROC={m_disc['all']['auroc']:.3f} | "
              f"mse AUROC={m_mse['all']['auroc']:.3f}")
        if "rpo_real_only" in m_at:
            print(f"   rpo_real_only AT-score AUROC={m_at['rpo_real_only']['auroc']:.3f} | "
                  f"disc AUROC={m_disc['rpo_real_only']['auroc']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
