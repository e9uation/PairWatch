from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pairwatch.eval.lstm_ad import to_tensor_per_window
from pairwatch.eval.physics_baselines import (
    cw_template_scores, rr_swing_scores, ocsvm_scores, dtw_1nn_scores,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"

NEG_SOURCES = ["clean", "routine_maneuver", "starlink_constellation",
               "formation_flying"]
NEG_LABEL = {"clean": "clean", "routine_maneuver": "routine_man.",
             "starlink_constellation": "starlink",
             "formation_flying": "formation_fly."}



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=REPO_ROOT / "runs" / "physics_per_source.json")
    args = ap.parse_args()

    meta = pd.read_parquet(BENCH_DIR / "windows_meta.parquet") \
             .sort_values("window_id").reset_index(drop=True)
    feat = pd.read_parquet(BENCH_DIR / "windows_feat.parquet")
    X, y_all = to_tensor_per_window(meta, feat)
    X_rate = X[:, :, 1:]
    train_idx = np.where((meta["split"] == "train").values)[0]
    test_idx = np.where((meta["split"] == "test").values)[0]

    print("scoring ...")
    scores = {
        "Range-rate swing": rr_swing_scores(X_rate[:, :, 0]),
        "CW template": cw_template_scores(X_rate),
        "DTW 1-NN": dtw_1nn_scores(X_rate, train_idx),
        "OC-SVM (pair feats)": ocsvm_scores(X_rate, train_idx),
    }

    src = meta["source"].str.split(":").str[0].values[test_idx]
    y = y_all[test_idx]

    rows = []
    for pos_label, pos_name in [(2, "real"), (3, "synth")]:
        for neg in NEG_SOURCES:
            pos_mask = (y == pos_label)
            neg_mask = (src == neg)
            entry = {"pos": pos_name, "neg": NEG_LABEL[neg],
                     "n_pos": int(pos_mask.sum()), "n_neg": int(neg_mask.sum())}
            for name, s in scores.items():
                st = s[test_idx]
                yy = np.r_[np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())]
                ss = np.r_[st[pos_mask], st[neg_mask]]
                entry[name] = float(roc_auc_score(yy, ss))
            rows.append(entry)

    names = list(scores)
    hdr = f"{'pos':6s} {'neg':15s} " + " ".join(f"{n:>19s}" for n in names)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['pos']:6s} {r['neg']:15s} " +
              " ".join(f"{r[n]:>19.2f}" for n in names))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
