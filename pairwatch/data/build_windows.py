from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pairwatch.data.pair_features import (
    compute_pair_series, load_tle_history,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw_tle"
CACHE_DIR = REPO_ROOT / "data" / "cache"
BENCH_DIR = REPO_ROOT / "benchmark"

WINDOW_HOURS = 14 * 24
STEP_S = 3600
COHORT = [44797, 44835, 42798, 42919, 42986, 53323]
NAMES = {44797: "Cosmos 2542", 44835: "Cosmos 2543",
         42798: "Cosmos 2519", 42919: "Cosmos 2521",
         42986: "Cosmos 2523", 53323: "Cosmos 2558"}


@dataclass
class Window:
    window_id: str
    target_id: int
    neighbor_id: int
    t_start: pd.Timestamp
    t_end: pd.Timestamp
    label: int
    source: str
    split: str


def time_split(t_start: pd.Timestamp, label: int = -1) -> str:
    if label == 2:
        return "test"
    if label == 3:
        return "val" if t_start.year <= 2020 else "test"
    y = t_start.year
    if y <= 2019:
        return "train"
    if y == 2020:
        return "val"
    return "test"


def make_window(target_id: int, neighbor_id: int,
                t_start: pd.Timestamp, label: int, source: str) -> Window:
    t_end = t_start + pd.Timedelta(hours=WINDOW_HOURS)
    return Window(
        window_id=str(uuid.uuid4())[:12],
        target_id=int(target_id), neighbor_id=int(neighbor_id),
        t_start=t_start, t_end=t_end,
        label=int(label), source=source,
        split=time_split(t_start, label),
    )


def overlap(a_start: pd.Timestamp, a_end: pd.Timestamp,
            b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def tle_coverage(norad: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    p = RAW_DIR / f"{norad}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df["epoch"].min(), df["epoch"].max()


def build_rpo_real(episodes: list[dict]) -> list[Window]:
    out = []
    for ep in episodes:
        src = ep["source"]
        if src.startswith("pair_"):
            a, b = src.replace("pair_", "").split("_")
        elif src == "cosmos2542_2543":
            a, b = "44797", "44835"
        else:
            continue
        t0 = pd.Timestamp(ep["t_start"])
        t1 = pd.Timestamp(ep["t_end"])
        midpoint = t0 + (t1 - t0) / 2
        win_start = midpoint - pd.Timedelta(hours=WINDOW_HOURS / 2)
        out.append(make_window(int(a), int(b), win_start, 2, "rpo_real"))
    return out


def build_routine_maneuver(maneuvers: pd.DataFrame, rng: np.random.Generator,
                           rpo_intervals: dict[int, list[tuple]]) -> list[Window]:
    out = []
    for _, row in maneuvers.iterrows():
        S = int(row["norad_id"])
        ep = row["epoch"]
        if not isinstance(ep, pd.Timestamp):
            ep = pd.Timestamp(ep)
        t_start = ep - pd.Timedelta(hours=WINDOW_HOURS / 2)
        t_end = ep + pd.Timedelta(hours=WINDOW_HOURS / 2)

        candidates = []
        for N in COHORT:
            if N == S:
                continue
            cov = tle_coverage(N)
            if cov is None or cov[0] > t_start or cov[1] < t_end:
                continue
            forbidden = rpo_intervals.get(tuple(sorted((S, N))), [])
            if any(overlap(t_start, t_end, fa, fb) for fa, fb in forbidden):
                continue
            candidates.append(N)
        if not candidates:
            continue
        N = int(rng.choice(candidates))
        out.append(make_window(S, N, t_start, 1, "routine_maneuver"))
    return out


def build_clean(rng: np.random.Generator, n_per_sat: int,
                rpo_intervals: dict[tuple, list[tuple]],
                man_epochs: dict[int, list[pd.Timestamp]]) -> list[Window]:
    out = []
    for S in COHORT:
        cov = tle_coverage(S)
        if cov is None:
            continue
        t_lo = cov[0] + pd.Timedelta(hours=WINDOW_HOURS)
        t_hi = cov[1] - pd.Timedelta(hours=WINDOW_HOURS)
        if t_hi <= t_lo:
            continue
        attempts = 0
        emitted = 0
        while emitted < n_per_sat and attempts < n_per_sat * 20:
            attempts += 1
            offset_s = rng.uniform(0, (t_hi - t_lo).total_seconds())
            t_start = t_lo + pd.Timedelta(seconds=float(offset_s))
            t_end = t_start + pd.Timedelta(hours=WINDOW_HOURS)
            if any(t_start <= m <= t_end for m in man_epochs.get(S, [])):
                continue
            bad = False
            for N in COHORT:
                if N == S:
                    continue
                key = tuple(sorted((S, N)))
                for fa, fb in rpo_intervals.get(key, []):
                    if overlap(t_start, t_end, fa, fb):
                        bad = True; break
                if bad: break
            if bad:
                continue
            neighbors = [N for N in COHORT if N != S and
                         (cov2 := tle_coverage(N)) is not None
                         and cov2[0] <= t_start and cov2[1] >= t_end]
            if not neighbors:
                continue
            N = int(rng.choice(neighbors))
            out.append(make_window(S, N, t_start, 0, "clean"))
            emitted += 1
    return out


def build_rpo_synth(synth_paths: list[Path]) -> tuple[list[Window], pd.DataFrame]:
    wins: list[Window] = []
    feats_rows: list[pd.DataFrame] = []
    for path in synth_paths:
        try:
            tgt = int(path.stem.replace("synth_rpo_", ""))
        except ValueError:
            tgt = 44797   # fallback
        df = pd.read_parquet(path)
        for ep_id, group in df.groupby("episode_id"):
            group = group.sort_values("t").reset_index(drop=True)
            if len(group) < 24:
                continue
            t_start = pd.Timestamp(group["t"].iloc[0]).tz_convert("UTC") \
                if pd.Timestamp(group["t"].iloc[0]).tz else pd.Timestamp(group["t"].iloc[0]).tz_localize("UTC")
            wid = make_window(tgt, tgt, t_start, 3,
                              f"rpo_synth:{path.stem}:{ep_id}")
            wid.neighbor_id = -1
            wins.append(wid)
            feat = group[["t", "range_km", "range_rate_km_s",
                          "rel_speed_km_s", "plane_angle_deg"]].copy()
            feat.insert(0, "window_id", wid.window_id)
            feats_rows.append(feat)
    feats = pd.concat(feats_rows, ignore_index=True) if feats_rows else pd.DataFrame()
    return wins, feats


def compute_features_for_real(wins: list[Window]) -> pd.DataFrame:
    rows = []
    for w in wins:
        try:
            att = load_tle_history(w.target_id)
            tgt = load_tle_history(w.neighbor_id)
        except FileNotFoundError:
            continue
        df = compute_pair_series(att, tgt,
                                 w.t_start.to_pydatetime(),
                                 w.t_end.to_pydatetime(),
                                 STEP_S)
        if df.empty:
            continue
        df = df[["t", "range_km", "range_rate_km_s",
                 "rel_speed_km_s", "plane_angle_deg"]].copy()
        df.insert(0, "window_id", w.window_id)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default=CACHE_DIR / "episodes.json")
    ap.add_argument("--maneuvers", default=CACHE_DIR / "maneuvers.csv")
    ap.add_argument("--synth", nargs="*",
                    default=[str(CACHE_DIR / "synth_rpo_44797.parquet")])
    ap.add_argument("--n-clean-per-sat", type=int, default=15)
    ap.add_argument("--n-routine-cap", type=int, default=60,
                    help="Subsample maneuvers to at most this many.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=BENCH_DIR)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    episodes = json.loads(Path(args.episodes).read_text())["episodes"]
    maneuvers = pd.read_csv(args.maneuvers, parse_dates=["epoch"])

    rpo_intervals: dict[tuple, list[tuple]] = {}
    for ep in episodes:
        src = ep["source"]
        if src.startswith("pair_"):
            a, b = src.replace("pair_", "").split("_")
        elif src == "cosmos2542_2543":
            a, b = "44797", "44835"
        else:
            continue
        key = tuple(sorted((int(a), int(b))))
        rpo_intervals.setdefault(key, []).append(
            (pd.Timestamp(ep["t_start"]), pd.Timestamp(ep["t_end"]))
        )
    man_epochs: dict[int, list[pd.Timestamp]] = (
        maneuvers.groupby("norad_id")["epoch"].apply(list).to_dict()
    )

    rpo_real_wins = build_rpo_real(episodes)

    if len(maneuvers) > args.n_routine_cap:
        maneuvers = maneuvers.sample(args.n_routine_cap, random_state=args.seed)
    routine_wins = build_routine_maneuver(maneuvers, rng, rpo_intervals)

    clean_wins = build_clean(rng, args.n_clean_per_sat, rpo_intervals, man_epochs)

    synth_paths = [Path(p) for p in args.synth]
    rpo_synth_wins, synth_feats = build_rpo_synth(synth_paths)

    all_wins = rpo_real_wins + routine_wins + clean_wins + rpo_synth_wins
    print(f"windows: rpo_real={len(rpo_real_wins)} "
          f"routine={len(routine_wins)} clean={len(clean_wins)} "
          f"rpo_synth={len(rpo_synth_wins)}  total={len(all_wins)}")

    real_wins = [w for w in all_wins if w.source != "rpo_synth" and
                 not w.source.startswith("rpo_synth")]
    real_feats = compute_features_for_real(real_wins)
    feats = pd.concat([real_feats, synth_feats], ignore_index=True) \
        if not synth_feats.empty else real_feats

    valid_ids = set(feats["window_id"].unique())
    all_wins = [w for w in all_wins if w.window_id in valid_ids]

    meta_df = pd.DataFrame([asdict(w) for w in all_wins])
    meta_df["t_start"] = pd.to_datetime(meta_df["t_start"], utc=True)
    meta_df["t_end"] = pd.to_datetime(meta_df["t_end"], utc=True)

    feats["t"] = pd.to_datetime(feats["t"], utc=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(out_dir / "windows_meta.parquet", index=False)
    feats.to_parquet(out_dir / "windows_feat.parquet", index=False)

    counts = (meta_df.groupby(["split", "label"]).size()
              .unstack(fill_value=0).astype(int))
    print("\n--- counts (rows = split, cols = label) ---")
    print(counts)
    stats = {
        "n_windows": int(len(meta_df)),
        "n_feature_rows": int(len(feats)),
        "by_split_label": counts.to_dict(),
        "by_source": meta_df["source"].apply(
            lambda s: s.split(":")[0]).value_counts().to_dict(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"\nWrote -> {out_dir}/")


if __name__ == "__main__":
    main()
