from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw_tle"
MU = 398600.4418


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--norad", nargs="+", type=int, required=True)
    p.add_argument("--threshold", type=float, default=6.0,
                   help="MAD multiplier for maneuver detection.")
    p.add_argument("--min-gap-h", type=float, default=12.0,
                   help="Suppress duplicate detections within this window.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def sma_from_tle(line1: str, line2: str) -> float:
    sat = Satrec.twoline2rv(line1, line2)
    n_rad_s = sat.no_kozai / 60.0
    return (MU / n_rad_s ** 2) ** (1 / 3)


def detect_for_one(df: pd.DataFrame, threshold: float,
                   min_gap_h: float) -> pd.DataFrame:
    df = df.sort_values("epoch").reset_index(drop=True)
    a = np.array([sma_from_tle(l1, l2)
                  for l1, l2 in zip(df["tle_line1"], df["tle_line2"])])
    dt_h = np.diff(df["epoch"].astype("int64").values) / 3.6e12
    da = np.diff(a)
    da_per_h = np.divide(da, dt_h, out=np.zeros_like(da), where=dt_h > 0)

    mad = np.median(np.abs(da_per_h - np.median(da_per_h))) + 1e-9
    flags = np.abs(da_per_h - np.median(da_per_h)) > threshold * 1.4826 * mad

    rows = []
    last_epoch = None
    for i, flag in enumerate(flags):
        if not flag:
            continue
        ep = df["epoch"].iloc[i + 1]
        if last_epoch is not None and (ep - last_epoch).total_seconds() < min_gap_h * 3600:
            continue
        rows.append({
            "norad_id": int(df["norad_id"].iloc[0]),
            "epoch": ep,
            "delta_sma_km": float(da[i]),
            "dt_h": float(dt_h[i]),
            "delta_sma_per_h": float(da_per_h[i]),
            "z_score": float(abs(da_per_h[i] - np.median(da_per_h))
                              / (1.4826 * mad)),
        })
        last_epoch = ep
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    all_rows = []
    for nid in args.norad:
        path = RAW_DIR / f"{nid}.parquet"
        if not path.exists():
            print(f"skip {nid}: no TLE file")
            continue
        df = pd.read_parquet(path)
        det = detect_for_one(df, args.threshold, args.min_gap_h)
        print(f"{nid}: {len(det):4d} maneuvers detected "
              f"(over {len(df)} TLEs, "
              f"{(df['epoch'].max() - df['epoch'].min()).days} days)")
        all_rows.append(det)
    if not all_rows:
        raise SystemExit("No detections.")
    out = pd.concat(all_rows, ignore_index=True).sort_values(["norad_id", "epoch"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nTotal {len(out)} maneuver events -> {args.out}")
    print(f"Per satellite: \n{out.groupby('norad_id').size()}")


if __name__ == "__main__":
    main()
