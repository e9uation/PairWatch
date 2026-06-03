from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from pairwatch.eval.lstm_ad import (
    SEQ_LEN, to_tensor_per_window, LSTMPredictor,
)
from pairwatch.eval.tran_ad import TranADLite
from pairwatch.eval.anomaly_transformer import (
    AnomalyTransformer, assoc_discrepancy as at_disc,
)
from pairwatch.eval.pairwatch import PairWatch, assoc_discrepancy as pw_disc

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default=REPO_ROOT / "runs" / "significance.json")
    return ap.parse_args()


def train_and_score(method: str, seed: int, Xt: torch.Tensor,
                    train_idx: np.ndarray, epochs: int) -> np.ndarray:
    torch.manual_seed(seed); np.random.seed(seed)
    in_dim = Xt.shape[2]
    loader = DataLoader(TensorDataset(Xt[train_idx]),
                        batch_size=8, shuffle=True)

    if method == "lstm_ad":
        model = LSTMPredictor(in_dim=in_dim, hidden=64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        inp = Xt[:, :-1, :]; tgt = Xt[:, 1:, :]
        for ep in range(epochs):
            for (xb,) in loader:
                ib = xb[:, :-1, :]; tb = xb[:, 1:, :]
                pred = model(ib)
                loss = F.mse_loss(pred, tb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred_all = model(inp)
            per_step = F.mse_loss(pred_all, tgt, reduction="none").mean(dim=2)
            return per_step.mean(dim=1).cpu().numpy()

    if method == "tran_ad":
        model = TranADLite(in_dim=in_dim, d_model=64, nhead=4, layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epochs):
            for (xb,) in loader:
                recon = model(xb)
                loss = F.mse_loss(recon, xb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            recon = model(Xt)
            per_step = F.mse_loss(recon, Xt, reduction="none").mean(dim=2)
            return per_step.mean(dim=1).cpu().numpy()

    if method == "anomtrans":
        model = AnomalyTransformer(in_dim=in_dim, d_model=64, nhead=4, layers=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epochs):
            for (xb,) in loader:
                recon, a, p = model(xb)
                loss = F.mse_loss(recon, xb) + 3.0 * at_disc(a, p).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            recon, a, p = model(Xt)
            per_step = F.mse_loss(recon, Xt, reduction="none").mean(dim=2)
            disc = at_disc(a, p)
            w = torch.softmax(-disc, dim=-1)
            return (w * per_step).sum(dim=-1).cpu().numpy()

    if method == "pairwatch":
        model = PairWatch(in_dim=in_dim, d_model=64, nhead=4, layers=2,
                           ablate="fixed_alpha")
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(epochs):
            for (xb,) in loader:
                recon, a, p = model(xb)
                loss = F.mse_loss(recon, xb) + 3.0 * pw_disc(a, p).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            recon, a, p = model(Xt)
            per_step = F.mse_loss(recon, Xt, reduction="none").mean(dim=2)
            disc = pw_disc(a, p)
            w = torch.softmax(-disc, dim=-1)
            return (w * per_step).sum(dim=-1).cpu().numpy()

    raise ValueError(method)


def bootstrap_auroc(y: np.ndarray, s: np.ndarray, n_boot: int,
                    rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(y)
    aurocs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            aurocs[b] = np.nan
            continue
        aurocs[b] = roc_auc_score(y[idx], s[idx])
    aurocs = aurocs[~np.isnan(aurocs)]
    return (float(roc_auc_score(y, s)),
            float(np.percentile(aurocs, 2.5)),
            float(np.percentile(aurocs, 97.5)))


def paired_bootstrap_diff(y: np.ndarray, sA: np.ndarray, sB: np.ndarray,
                           n_boot: int, rng: np.random.Generator) -> dict:
    n = len(y)
    diffs = np.empty(n_boot)
    a_aurocs = np.empty(n_boot)
    b_aurocs = np.empty(n_boot)
    valid = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            diffs[b] = np.nan; continue
        try:
            a = roc_auc_score(y[idx], sA[idx])
            b_ = roc_auc_score(y[idx], sB[idx])
        except ValueError:
            diffs[b] = np.nan; continue
        diffs[b] = a - b_
        a_aurocs[b] = a; b_aurocs[b] = b_
        valid += 1
    diffs = diffs[~np.isnan(diffs)]
    point_a = roc_auc_score(y, sA)
    point_b = roc_auc_score(y, sB)
    p_two_sided = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "auroc_A": float(point_a),
        "auroc_B": float(point_b),
        "diff_mean": float(diffs.mean()),
        "diff_ci_lo": float(np.percentile(diffs, 2.5)),
        "diff_ci_hi": float(np.percentile(diffs, 97.5)),
        "p_value_boot": float(p_two_sided),
        "n_valid": int(valid),
    }


def delong_test(y: np.ndarray, sA: np.ndarray, sB: np.ndarray) -> dict:
    from scipy.stats import norm

    def midrank(x):
        J = np.argsort(x)
        Z = x[J]
        n = len(x)
        T = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        T2 = np.empty(n)
        T2[J] = T
        return T2

    def fast_auc_cov(y, scores):
        m = int(y.sum()); nn = len(y) - m
        pos_scores = np.column_stack([s[y == 1] for s in scores])
        neg_scores = np.column_stack([s[y == 0] for s in scores])
        k = len(scores)
        tx = np.zeros((k, m)); ty = np.zeros((k, nn)); tz = np.zeros((k, m + nn))
        aucs = np.empty(k)
        for r in range(k):
            tx[r] = midrank(pos_scores[:, r])
            ty[r] = midrank(neg_scores[:, r])
            tz[r] = midrank(np.r_[pos_scores[:, r], neg_scores[:, r]])
            aucs[r] = (tz[r, :m].sum() / (m * nn)) - (m + 1) / (2.0 * nn)
        v01 = (tz[:, :m] - tx) / nn
        v10 = 1.0 - (tz[:, m:] - ty) / m
        sx = np.cov(v01); sy = np.cov(v10)
        if k == 1:
            sx = np.array([[sx]]); sy = np.array([[sy]])
        cov = sx / m + sy / nn
        return aucs, cov

    aucs, cov = fast_auc_cov(y, [sA, sB])
    L = np.array([1, -1])
    var = L @ cov @ L
    if var <= 0:
        return {"auroc_A": float(aucs[0]), "auroc_B": float(aucs[1]),
                "z": float("nan"), "p_value": float("nan")}
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"auroc_A": float(aucs[0]), "auroc_B": float(aucs[1]),
            "delong_z": float(z), "delong_p": float(p),
            "var_diff": float(var)}


def main() -> None:
    args = parse_args()
    print("loading benchmark...")
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
    y_test = (np.isin(y_all[test_idx], [2, 3])).astype(int)
    y_test_rpo_real = y_all[test_idx]

    methods = ["lstm_ad", "tran_ad", "anomtrans", "pairwatch"]
    scores: dict[str, list[np.ndarray]] = {m: [] for m in methods}

    for seed in args.seeds:
        for m in methods:
            print(f"train {m} seed={seed}...")
            s_all = train_and_score(m, seed, Xt, train_idx, args.epochs)
            scores[m].append(s_all[test_idx])

    scores_mean = {m: np.mean(np.stack(scores[m]), axis=0) for m in methods}

    rng = np.random.default_rng(0)
    result = {"seeds": args.seeds, "n_boot": args.boot,
              "n_test": int(test_mask.sum()),
              "n_pos_test": int(y_test.sum())}

    print("\n--- bootstrap CI: test all ---")
    boot = {}
    for m in methods:
        pt, lo, hi = bootstrap_auroc(y_test, scores_mean[m], args.boot, rng)
        boot[m] = {"auroc": pt, "ci_lo": lo, "ci_hi": hi}
        print(f"  {m:12s}  AUROC={pt:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")
    result["bootstrap_test_all"] = boot

    neg_mask = np.isin(y_test_rpo_real, [0, 1])
    sub_mask = neg_mask | (y_test_rpo_real == 2)
    y_sub = (y_test_rpo_real[sub_mask] == 2).astype(int)
    print(f"\n--- bootstrap CI: rpo_real subset (n={sub_mask.sum()}, pos={y_sub.sum()}) ---")
    boot_real = {}
    for m in methods:
        pt, lo, hi = bootstrap_auroc(y_sub, scores_mean[m][sub_mask],
                                     args.boot, rng)
        boot_real[m] = {"auroc": pt, "ci_lo": lo, "ci_hi": hi}
        print(f"  {m:12s}  AUROC={pt:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")
    result["bootstrap_rpo_real"] = boot_real

    print("\n--- paired bootstrap CI: PairWatch - baseline (test all) ---")
    pw = scores_mean["pairwatch"]
    diffs = {}
    for m in ["lstm_ad", "tran_ad", "anomtrans"]:
        d = paired_bootstrap_diff(y_test, pw, scores_mean[m], args.boot, rng)
        try:
            dl = delong_test(y_test, pw, scores_mean[m])
            d.update(dl)
        except Exception as e:
            d["delong_error"] = str(e)
        diffs[m] = d
        print(f"  vs {m:12s}: diff={d['diff_mean']:+.4f} "
              f"95%CI=[{d['diff_ci_lo']:+.4f}, {d['diff_ci_hi']:+.4f}]  "
              f"boot p={d['p_value_boot']:.3g}  "
              f"DeLong p={d.get('delong_p', float('nan')):.3g}")
    result["paired_test_all"] = diffs

    print("\n--- paired bootstrap CI: PairWatch - baseline (rpo_real) ---")
    pw_sub = pw[sub_mask]
    diffs_real = {}
    for m in ["lstm_ad", "tran_ad", "anomtrans"]:
        d = paired_bootstrap_diff(y_sub, pw_sub, scores_mean[m][sub_mask],
                                   args.boot, rng)
        try:
            dl = delong_test(y_sub, pw_sub, scores_mean[m][sub_mask])
            d.update(dl)
        except Exception as e:
            d["delong_error"] = str(e)
        diffs_real[m] = d
        print(f"  vs {m:12s}: diff={d['diff_mean']:+.4f} "
              f"95%CI=[{d['diff_ci_lo']:+.4f}, {d['diff_ci_hi']:+.4f}]  "
              f"boot p={d['p_value_boot']:.3g}  "
              f"DeLong p={d.get('delong_p', float('nan')):.3g}")
    result["paired_rpo_real"] = diffs_real

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote -> {args.out}")


if __name__ == "__main__":
    main()
