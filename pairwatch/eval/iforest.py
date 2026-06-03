from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec
from sklearn.ensemble import IsolationForest

from pairwatch.eval.metrics import compute_metrics, stratified_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw_tle"
BENCH_DIR = REPO_ROOT / "benchmark"
MU = 398600.4418


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default=BENCH_DIR / "windows_meta.parquet")
    p.add_argument("--out", default=REPO_ROOT / "runs" / "iforest.json")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def sma_km_from_lines(l1: str, l2: str) -> float:
    sat = Satrec.twoline2rv(l1, l2)
    n_rad_s = sat.no_kozai / 60.0
    return (MU / n_rad_s ** 2) ** (1 / 3)


def window_features(target_id: int, t_start: pd.Timestamp,
                    t_end: pd.Timestamp) -> np.ndarray:
    p = RAW_DIR / f"{target_id}.parquet"
    if target_id < 0 or not p.exists():
        return np.zeros(7, dtype=float)
    df = pd.read_parquet(p)
    mask = (df["epoch"] >= t_start) & (df["epoch"] <= t_end)
    sub = df.loc[mask].sort_values("epoch")
    if len(sub) < 3:
        return np.zeros(7, dtype=float)
    smas = np.array([sma_km_from_lines(l1, l2)
                     for l1, l2 in zip(sub["tle_line1"], sub["tle_line2"])])
    t_s = (sub["epoch"].astype("int64").to_numpy() - sub["epoch"].astype("int64").iloc[0]) / 1e9
    A = np.column_stack([t_s, np.ones_like(t_s)])
    coef, *_ = np.linalg.lstsq(A, smas, rcond=None)
    resid = smas - A @ coef
    return np.array([
        float(np.std(resid)),
        float(np.max(np.abs(resid))),
        float(np.percentile(np.abs(resid), 90)),
        float(np.sum(np.abs(resid) > 0.1)),
        float(np.sum(np.abs(resid) > 1.0)),
        float(coef[0]),
        float(np.mean(smas)),
    ], dtype=float)


def main() -> None:
    args = parse_args()
    meta = pd.read_parquet(args.meta)
    meta["t_start"] = pd.to_datetime(meta["t_start"], utc=True)
    meta["t_end"] = pd.to_datetime(meta["t_end"], utc=True)

    print(f"Building features for {len(meta)} windows...")
    feats = np.stack([window_features(int(r.target_id), r.t_start, r.t_end)
                      for r in meta.itertuples(index=False)])
    print(f"feature matrix: {feats.shape}")

    train_mask = (meta["split"] == "train").values
    val_mask = (meta["split"] == "val").values
    test_mask = (meta["split"] == "test").values

    train_X = feats[train_mask]
    if len(train_X) < 5:
        raise SystemExit("Train split too small for Isolation Forest.")

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination="auto",
        random_state=args.seed,
    ).fit(train_X)

    scores_all = -model.score_samples(feats)

    out = {
        "model": "isolation_forest_sma_residual",
        "n_train": int(train_mask.sum()),
        "feature_dim": feats.shape[1],
    }
    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        if mask.sum() == 0:
            continue
        m = stratified_metrics(meta.loc[mask, "label"].to_numpy(),
                               scores_all[mask])
        out[split_name] = m
        all_m = m["all"]
        print(f"{split_name}: n={all_m['n']} pos={all_m['n_pos']} "
              f"AUROC={all_m['auroc']:.3f}  AP={all_m['ap']:.3f}  "
              f"F1={all_m['f1']:.3f}  P@1%FPR={all_m['p_at_1pct_fpr']:.3f}")

    args_out = Path(args.out)
    args_out.parent.mkdir(parents=True, exist_ok=True)
    args_out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote -> {args_out}")


if __name__ == "__main__":
    main()
