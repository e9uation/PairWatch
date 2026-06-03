from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw_tle"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--attacker", type=int, required=True)
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--start", type=datetime.fromisoformat, required=True)
    p.add_argument("--end", type=datetime.fromisoformat, required=True)
    p.add_argument("--step", type=int, default=3600,
                   help="Propagation step in seconds (default 1h).")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def load_tle_history(norad: int) -> pd.DataFrame:
    path = RAW_DIR / f"{norad}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run fetch_tle for {norad} first ({path})")
    return pd.read_parquet(path)


def select_tle(df: pd.DataFrame, t: datetime) -> tuple[str, str]:
    t_utc = pd.Timestamp(t, tz="UTC") if t.tzinfo is None else pd.Timestamp(t)
    deltas = (df["epoch"] - t_utc).abs()
    idx = deltas.idxmin()
    return df.loc[idx, "tle_line1"], df.loc[idx, "tle_line2"]


def propagate_state(tle1: str, tle2: str, t: datetime) -> tuple[np.ndarray, np.ndarray]:
    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(t.year, t.month, t.day,
                  t.hour, t.minute, t.second + t.microsecond * 1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return np.full(3, np.nan), np.full(3, np.nan)
    return np.asarray(r), np.asarray(v)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    cos = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(cos))


def compute_pair_series(att_df: pd.DataFrame, tgt_df: pd.DataFrame,
                        start: datetime, end: datetime,
                        step_s: int) -> pd.DataFrame:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
        end = end.replace(tzinfo=timezone.utc)
    times = pd.date_range(start, end, freq=f"{step_s}s", tz="UTC")

    rows = []
    a_l1, a_l2 = None, None
    t_l1, t_l2 = None, None
    last_a_epoch = None
    last_t_epoch = None

    for t in times:
        py_t = t.to_pydatetime()
        if last_a_epoch is None or (py_t - last_a_epoch).total_seconds() > 86400:
            a_l1, a_l2 = select_tle(att_df, py_t)
            last_a_epoch = py_t
        if last_t_epoch is None or (py_t - last_t_epoch).total_seconds() > 86400:
            t_l1, t_l2 = select_tle(tgt_df, py_t)
            last_t_epoch = py_t

        ra, va = propagate_state(a_l1, a_l2, py_t)
        rt, vt = propagate_state(t_l1, t_l2, py_t)
        if np.any(np.isnan(ra)) or np.any(np.isnan(rt)):
            continue

        rel_r = ra - rt
        rel_v = va - vt
        rng = float(np.linalg.norm(rel_r))
        rng_rate = float(np.dot(rel_r, rel_v) / rng) if rng > 0 else float("nan")
        n_a = np.cross(ra, va)
        n_t = np.cross(rt, vt)
        plane_angle_deg = np.degrees(angle_between(n_a, n_t))

        rows.append({
            "t": t, "range_km": rng, "range_rate_km_s": rng_rate,
            "rel_speed_km_s": float(np.linalg.norm(rel_v)),
            "plane_angle_deg": plane_angle_deg,
        })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    att = load_tle_history(args.attacker)
    tgt = load_tle_history(args.target)
    print(f"Attacker {args.attacker}: {len(att)} TLEs")
    print(f"Target   {args.target}: {len(tgt)} TLEs")

    series = compute_pair_series(att, tgt, args.start, args.end, args.step)
    if series.empty:
        print("No valid samples produced.")
        return

    idx_min = series["range_km"].idxmin()
    print(f"Min range  = {series.loc[idx_min, 'range_km']:.2f} km "
          f"@ {series.loc[idx_min, 't']}")
    print(f"Median rng = {series['range_km'].median():.2f} km")
    print(f"Median |plane angle| = {series['plane_angle_deg'].abs().median():.2f} deg")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        series.to_parquet(args.out, index=False)
        print(f"Wrote {len(series)} rows -> {args.out}")


if __name__ == "__main__":
    main()
