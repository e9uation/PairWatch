from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pairwatch.data.build_windows import (
    Window, WINDOW_HOURS, STEP_S, time_split,
)
from pairwatch.data.pair_features import (
    compute_pair_series, load_tle_history,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"

PAIRS = [
    (43476, 43477),
    (27424, 28376),
    (31698, 36605),
    (43481, 43482),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-per-pair", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    meta = pd.read_parquet(BENCH_DIR / "windows_meta.parquet")
    feat = pd.read_parquet(BENCH_DIR / "windows_feat.parquet")
    new_meta_rows = []
    new_feat_frames = []
    pair_stats = []

    for a, b in PAIRS:
        try:
            df_a = load_tle_history(a)
            df_b = load_tle_history(b)
        except FileNotFoundError as e:
            print(f"skip {a}/{b}: {e}")
            continue
        t_lo = max(df_a["epoch"].min(), df_b["epoch"].min())
        t_hi = min(df_a["epoch"].max(), df_b["epoch"].max())
        span_s = (t_hi - t_lo).total_seconds() - WINDOW_HOURS * 3600
        if span_s <= 0:
            print(f"skip {a}/{b}: insufficient overlap")
            continue
        pair_min_ranges = []
        for _ in range(args.windows_per_pair):
            offset = rng.uniform(0, span_s)
            t_start = t_lo + pd.Timedelta(seconds=float(offset))
            t_end = t_start + pd.Timedelta(hours=WINDOW_HOURS)
            series = compute_pair_series(
                df_a, df_b, t_start.to_pydatetime(),
                t_end.to_pydatetime(), STEP_S)
            if series.empty:
                continue
            wid = str(uuid.uuid4())[:12]
            w = Window(
                window_id=wid, target_id=int(a), neighbor_id=int(b),
                t_start=t_start, t_end=t_end, label=0,
                source="formation_flying",
                split=time_split(t_start, label=0),
            )
            new_meta_rows.append(asdict(w))
            sub = series[["t", "range_km", "range_rate_km_s",
                          "rel_speed_km_s", "plane_angle_deg"]].copy()
            sub.insert(0, "window_id", wid)
            new_feat_frames.append(sub)
            pair_min_ranges.append(float(series["range_km"].min()))
        pair_stats.append({"pair": f"{a}_{b}", "n_windows_attempted":
                            args.windows_per_pair,
                            "min_range_observed": float(np.min(pair_min_ranges))
                            if pair_min_ranges else None,
                            "median_min_range": float(np.median(pair_min_ranges))
                            if pair_min_ranges else None})

    if not new_meta_rows:
        raise SystemExit("No formation windows produced.")

    new_meta = pd.DataFrame(new_meta_rows)
    new_meta["t_start"] = pd.to_datetime(new_meta["t_start"], utc=True)
    new_meta["t_end"] = pd.to_datetime(new_meta["t_end"], utc=True)
    meta_aug = pd.concat([meta, new_meta], ignore_index=True)
    feat_aug = pd.concat([feat, *new_feat_frames], ignore_index=True)
    feat_aug["t"] = pd.to_datetime(feat_aug["t"], utc=True)

    meta_aug.to_parquet(BENCH_DIR / "windows_meta.parquet", index=False)
    feat_aug.to_parquet(BENCH_DIR / "windows_feat.parquet", index=False)

    print(f"\nAdded {len(new_meta_rows)} formation-flying hard-negative windows.")
    for s in pair_stats:
        print(f"  {s['pair']:14s}  median_min_range={s['median_min_range']:.1f} km")
    counts = (meta_aug.groupby(["split", "label"]).size()
              .unstack(fill_value=0).astype(int))
    print("\n--- new totals: split x label ---")
    print(counts)
    fmt_feat = pd.concat(new_feat_frames, ignore_index=True)
    print("\n--- formation range distribution (km) ---")
    print(fmt_feat["range_km"].describe().round(2))


if __name__ == "__main__":
    main()
