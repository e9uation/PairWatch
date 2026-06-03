from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--norad", nargs="+", type=int, required=True)
    p.add_argument("--start", type=date.fromisoformat, required=True)
    p.add_argument("--end", type=date.fromisoformat, required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw_tle")
    p.add_argument("--sleep", type=float, default=1.5,
                   help="Seconds between per-NORAD requests (rate limit).")
    return p.parse_args()


def get_client() -> SpaceTrackClient:
    load_dotenv(REPO_ROOT / ".env")
    user = os.environ.get("SPACETRACK_USER")
    pw = os.environ.get("SPACETRACK_PASS")
    if not user or not pw:
        sys.exit("Missing SPACETRACK_USER / SPACETRACK_PASS in .env")
    return SpaceTrackClient(identity=user, password=pw)


def _parse_tle_epoch(line1: str) -> datetime:
    yy = int(line1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    day_of_year_frac = float(line1[20:32])
    doy = int(day_of_year_frac)
    frac = day_of_year_frac - doy
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    return base + pd.Timedelta(days=doy - 1 + frac)


def _split_tle_block(text: str) -> list[tuple[str, str]]:
    pairs = []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i + 1 < len(lines):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            pairs.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1
    return pairs


def fetch_one(client: SpaceTrackClient, norad: int,
              start: date, end: date) -> pd.DataFrame:
    text = client.gp_history(
        norad_cat_id=norad,
        epoch=op.inclusive_range(start, end),
        orderby="epoch asc",
        format="tle",
    )
    if not isinstance(text, str):
        text = "".join(text)
    pairs = _split_tle_block(text)
    if not pairs:
        return pd.DataFrame()

    rows = []
    for l1, l2 in pairs:
        try:
            epoch = _parse_tle_epoch(l1)
            row_norad = int(l1[2:7])
        except Exception:
            continue
        rows.append({"norad_id": row_norad, "epoch": epoch,
                     "tle_line1": l1, "tle_line2": l2})
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["epoch"]).sort_values("epoch")
    return df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    client = get_client()

    for norad in tqdm(args.norad, desc="NORAD"):
        out_path = args.out / f"{norad}.parquet"
        if out_path.exists():
            tqdm.write(f"skip {norad} (cached: {out_path.name})")
            continue
        df = fetch_one(client, norad, args.start, args.end)
        if df.empty:
            tqdm.write(f"empty result for {norad}")
            continue
        df.to_parquet(out_path, index=False)
        tqdm.write(f"{norad}: {len(df)} TLEs "
                   f"[{df['epoch'].min()} -> {df['epoch'].max()}]")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
