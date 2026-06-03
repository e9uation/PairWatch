from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw_tle"
MU_EARTH_KM3_S2 = 398600.4418


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, required=True,
                   help="Target NORAD ID with cached TLE history.")
    p.add_argument("--episodes-json", type=Path, required=True)
    p.add_argument("--n", type=int, default=200,
                   help="Number of synthetic episodes to generate.")
    p.add_argument("--duration-h", type=float, default=168.0,
                   help="Window length per episode (hours).")
    p.add_argument("--step", type=int, default=3600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hard", action="store_true",
                   help="Fix D: generate sub-threshold (harder) RPO.")
    return p.parse_args()


def load_target_tle(norad: int) -> pd.DataFrame:
    p = RAW_DIR / f"{norad}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing TLE for {norad}: {p}")
    df = pd.read_parquet(p)
    df["epoch"] = pd.to_datetime(df["epoch"], utc=True)
    return df.sort_values("epoch").reset_index(drop=True)


def select_tle(df: pd.DataFrame, t: pd.Timestamp) -> tuple[str, str]:
    deltas = (df["epoch"] - t).abs()
    i = deltas.idxmin()
    return df.loc[i, "tle_line1"], df.loc[i, "tle_line2"]


def state_eci(tle1: str, tle2: str, t: datetime) -> tuple[np.ndarray, np.ndarray]:
    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(t.year, t.month, t.day,
                  t.hour, t.minute, t.second + t.microsecond * 1e-6)
    e, r, v = sat.sgp4(jd, fr)
    return (np.asarray(r), np.asarray(v)) if e == 0 \
        else (np.full(3, np.nan), np.full(3, np.nan))


def lvlh_basis(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    R = r / np.linalg.norm(r)
    h = np.cross(r, v)
    W = h / np.linalg.norm(h)
    S = np.cross(W, R)
    return np.column_stack([R, S, W])


def cw_football(A: float, B: float, C: float, phi: float,
                n: float, t: np.ndarray) -> np.ndarray:
    nt = n * t
    x = -A * np.cos(nt)
    y = 2.0 * A * np.sin(nt) + B
    z = C * np.sin(nt + phi)
    return np.column_stack([x, y, z])


def cw_velocity(A: float, B: float, C: float, phi: float,
                n: float, t: np.ndarray) -> np.ndarray:
    nt = n * t
    vx = A * n * np.sin(nt)
    vy = 2.0 * A * n * np.cos(nt)
    vz = C * n * np.cos(nt + phi)
    return np.column_stack([vx, vy, vz])


def sample_params(dist: dict, rng: np.random.Generator,
                   hard: bool = False) -> dict:
    def lognormal_from_quantiles(p10: float, p90: float) -> float:
        p10 = max(p10, 1e-3); p90 = max(p90, p10 * 1.01)
        mu = (np.log(p10) + np.log(p90)) / 2
        sigma = (np.log(p90) - np.log(p10)) / (2 * 1.2816)
        return float(np.exp(rng.normal(mu, sigma)))

    if hard:
        min_rng = lognormal_from_quantiles(50.0, 500.0)
        plane = lognormal_from_quantiles(0.01, 0.5)
        along = rng.uniform(-1, 1) * min_rng * rng.uniform(5.0, 15.0)
        cross = abs(np.deg2rad(plane)) * 7000.0 * rng.uniform(0.5, 2.0)
    else:
        min_rng = lognormal_from_quantiles(
            dist["min_range_km"]["p10"], dist["min_range_km"]["p90"])
        plane = lognormal_from_quantiles(
            max(dist["median_plane_angle_deg"]["p10"], 1e-3),
            max(dist["median_plane_angle_deg"]["p90"], 1e-2))
        along = rng.uniform(-1, 1) * min_rng * rng.uniform(2.0, 8.0)
        cross = abs(np.deg2rad(plane)) * 7000.0 * rng.uniform(0.3, 1.5)
    phi = rng.uniform(0, 2 * np.pi)
    return {"A": min_rng, "B": along, "C": cross, "phi": phi,
            "min_range_km": min_rng, "plane_angle_deg": plane}


def generate_episode(target_df: pd.DataFrame, params: dict,
                     start: datetime, duration_h: float,
                     step_s: int) -> pd.DataFrame:
    times = pd.date_range(
        start, start + pd.Timedelta(hours=duration_h),
        freq=f"{step_s}s", tz="UTC")
    t_seconds = np.array([(tt - times[0]).total_seconds() for tt in times])

    l1, l2 = select_tle(target_df, times[0])
    sat = Satrec.twoline2rv(l1, l2)
    a_km = (MU_EARTH_KM3_S2 / sat.no_kozai ** 2) ** (1 / 3) \
        if False else None
    n_rad_s = sat.no_kozai / 60.0

    rel_lvlh = cw_football(params["A"], params["B"], params["C"],
                            params["phi"], n_rad_s, t_seconds)
    rel_v_lvlh = cw_velocity(params["A"], params["B"], params["C"],
                              params["phi"], n_rad_s, t_seconds)

    rows = []
    refresh_every_s = 86400
    last_refresh = None
    cur_l1, cur_l2 = l1, l2

    for i, tt in enumerate(times):
        py_t = tt.to_pydatetime()
        if last_refresh is None or (py_t - last_refresh).total_seconds() > refresh_every_s:
            cur_l1, cur_l2 = select_tle(target_df, tt)
            last_refresh = py_t
        r_t, v_t = state_eci(cur_l1, cur_l2, py_t)
        if np.any(np.isnan(r_t)):
            continue
        Q = lvlh_basis(r_t, v_t)
        r_a = r_t + Q @ rel_lvlh[i]
        v_a = v_t + Q @ rel_v_lvlh[i]
        rel = r_a - r_t
        relv = v_a - v_t
        rng = float(np.linalg.norm(rel))
        rng_rate = float(np.dot(rel, relv) / rng) if rng > 0 else float("nan")
        n_a = np.cross(r_a, v_a); n_t = np.cross(r_t, v_t)
        cos = float(np.clip(np.dot(n_a, n_t) /
                            (np.linalg.norm(n_a) * np.linalg.norm(n_t)), -1, 1))
        plane = float(np.degrees(np.arccos(cos)))
        rows.append({
            "t": tt, "range_km": rng, "range_rate_km_s": rng_rate,
            "rel_speed_km_s": float(np.linalg.norm(relv)),
            "plane_angle_deg": plane,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    target_df = load_target_tle(args.target)
    dist = json.loads(args.episodes_json.read_text())["distributions"]

    t0_min = target_df["epoch"].min().to_pydatetime()
    t1_max = target_df["epoch"].max().to_pydatetime() \
        - pd.Timedelta(hours=args.duration_h).to_pytimedelta()
    if t1_max <= t0_min:
        raise SystemExit("Target TLE coverage shorter than requested duration.")
    span_s = (t1_max - t0_min).total_seconds()

    all_rows = []
    for ep_id in range(args.n):
        params = sample_params(dist, rng, hard=args.hard)
        offset_s = rng.uniform(0, span_s)
        start = t0_min + pd.Timedelta(seconds=float(offset_s)).to_pytimedelta()
        df = generate_episode(target_df, params, start,
                              args.duration_h, args.step)
        if df.empty:
            continue
        df.insert(0, "episode_id", ep_id)
        df["param_min_range_km"] = params["min_range_km"]
        df["param_plane_angle_deg"] = params["plane_angle_deg"]
        all_rows.append(df)

    if not all_rows:
        raise SystemExit("No synthetic episodes produced.")
    out = pd.concat(all_rows, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    n_eps = out["episode_id"].nunique()
    print(f"Wrote {len(out)} rows across {n_eps} episodes -> {args.out}")
    print(f"Realised min ranges: "
          f"p10={out.groupby('episode_id')['range_km'].min().quantile(0.1):.2f} "
          f"p50={out.groupby('episode_id')['range_km'].min().quantile(0.5):.2f} "
          f"p90={out.groupby('episode_id')['range_km'].min().quantile(0.9):.2f}")


if __name__ == "__main__":
    main()
