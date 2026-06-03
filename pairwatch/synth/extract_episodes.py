from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", required=True,
                   help="Glob(s) for pair-feature parquet files.")
    p.add_argument("--threshold", type=float, default=200.0,
                   help="Range threshold (km) defining an approach episode.")
    p.add_argument("--min-duration-h", type=float, default=6.0,
                   help="Minimum contiguous duration (hours) for an episode.")
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def find_episodes(df: pd.DataFrame, threshold_km: float,
                  min_duration_h: float) -> list[dict]:
    df = df.sort_values("t").reset_index(drop=True)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    inside = df["range_km"].values < threshold_km
    episodes = []
    if inside.sum() == 0:
        return episodes

    edges = np.diff(inside.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if inside[0]:
        starts = np.r_[0, starts]
    if inside[-1]:
        ends = np.r_[ends, len(inside)]

    for s, e in zip(starts, ends):
        seg = df.iloc[s:e]
        if len(seg) < 2:
            continue
        dur_h = (seg["t"].iloc[-1] - seg["t"].iloc[0]).total_seconds() / 3600
        if dur_h < min_duration_h:
            continue
        rng = seg["range_km"].values
        rng_rate = seg["range_rate_km_s"].values
        plane = np.abs(seg["plane_angle_deg"].values)

        i_min = int(rng.argmin())
        approach_rates = rng_rate[:i_min + 1]
        approach_rates = approach_rates[approach_rates < 0]
        closing_rate = float(np.median(approach_rates)) if approach_rates.size \
            else float("nan")

        episodes.append({
            "t_start": seg["t"].iloc[0].isoformat(),
            "t_end": seg["t"].iloc[-1].isoformat(),
            "duration_h": float(dur_h),
            "min_range_km": float(rng.min()),
            "median_range_km": float(np.median(rng)),
            "median_closing_rate_km_s": closing_rate,
            "median_plane_angle_deg": float(np.median(plane)),
            "median_rel_speed_km_s": float(np.median(seg["rel_speed_km_s"].values)),
        })
    return episodes


def summarize(all_eps: list[dict]) -> dict:
    if not all_eps:
        return {}
    df = pd.DataFrame(all_eps)
    out = {}
    for col in ["duration_h", "min_range_km", "median_closing_rate_km_s",
                "median_plane_angle_deg", "median_rel_speed_km_s"]:
        v = df[col].dropna().values
        if v.size == 0:
            continue
        out[col] = {
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }
    return out


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for pat in args.pairs:
        paths.extend(Path(p) for p in glob.glob(pat))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No pair files matched.")

    all_eps = []
    for p in paths:
        df = pd.read_parquet(p)
        eps = find_episodes(df, args.threshold, args.min_duration_h)
        for ep in eps:
            ep["source"] = p.stem
        all_eps.extend(eps)
        print(f"{p.name}: {len(eps)} episode(s) "
              f"(min range {df['range_km'].min():.2f} km)")

    summary = summarize(all_eps)
    payload = {
        "config": {"threshold_km": args.threshold,
                   "min_duration_h": args.min_duration_h,
                   "n_sources": len(paths)},
        "episodes": all_eps,
        "distributions": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nTotal episodes: {len(all_eps)}")
    if summary:
        print(f"Distribution keys: {list(summary)}")
    print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
