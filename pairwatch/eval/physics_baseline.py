from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from pairwatch.eval.lstm_ad import to_tensor_per_window
from pairwatch.eval.metrics import stratified_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmark"
OPERATIONAL_NEG_SOURCES = ["clean", "routine_maneuver", "starlink_constellation"]
N_BOOT = 1000
SEQ_LEN = 336
DTW_DOWNSAMPLE = 4
DTW_BAND = 10


def cw_template_bank(n_steps: int = SEQ_LEN) -> np.ndarray:
    t_h = np.arange(n_steps, dtype=float)
    templates = []

    for period_h in (1.5, 1.6, 12.0, 24.0):
        n_rate = 2.0 * np.pi / period_h
        for amp_km in (0.5, 5.0, 50.0):
            for standoff_km in (20.0, 200.0, 2000.0):
                for t_ca_frac in (0.35, 0.5, 0.65):
                    t_ca = t_ca_frac * n_steps
                    b_t = amp_km + (standoff_km - amp_km) * np.abs(
                        t_h - t_ca) / max(t_ca, n_steps - t_ca)
                    theta = n_rate * t_h
                    x = amp_km * np.sin(theta)
                    y = 2.0 * amp_km * np.cos(theta) + b_t
                    z = 0.5 * amp_km * np.sin(theta + 0.7)
                    pos = np.stack([x, y, z], axis=1)

                    vel = np.gradient(pos, axis=0) / 3600.0
                    rng = np.linalg.norm(pos, axis=1) + 1e-9
                    rng_rate = np.sum(pos * vel, axis=1) / rng
                    rel_speed = np.linalg.norm(vel, axis=1)
                    plane = np.degrees(np.arcsin(
                        np.clip(np.abs(z) / rng, 0.0, 1.0)))

                    tpl = np.stack([rng_rate, rel_speed, plane], axis=1)
                    templates.append(_zscore_cols(tpl))
    return np.stack(templates)


def _zscore_cols(a: np.ndarray) -> np.ndarray:
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True) + 1e-9
    return (a - mu) / sd


def cw_template_scores(X: np.ndarray, shifts: int = 12) -> np.ndarray:
    bank = cw_template_bank(X.shape[1])
    Xz = np.stack([_zscore_cols(x) for x in X])
    T = X.shape[1]
    shift_list = np.linspace(0, T - 1, shifts, dtype=int)
    best = np.full(len(X), -np.inf)
    for tpl in bank:
        for s in shift_list:
            rolled = np.roll(tpl, s, axis=0)
            corr = np.einsum("ntc,tc->n", Xz, rolled) / (T * Xz.shape[2])
            best = np.maximum(best, corr)
    return best


def rr_swing_scores(rr: np.ndarray) -> np.ndarray:
    N, T = rr.shape
    csum = np.concatenate([np.zeros((N, 1)), np.cumsum(rr, axis=1)], axis=1)
    ks = np.arange(1, T)
    before = csum[:, ks] / ks                       # mean of rr[:k]
    after = (csum[:, T][:, None] - csum[:, ks]) / (T - ks)
    return np.max(after - before, axis=1)


def _dtw_band(a: np.ndarray, b: np.ndarray, band: int = DTW_BAND) -> float:
    n, m = len(a), len(b)
    INF = np.inf
    prev = np.full(m + 1, INF)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, INF)
        lo = max(1, i - band)
        hi = min(m, i + band)
        d = np.sum((a[i - 1] - b[lo - 1:hi]) ** 2, axis=1)
        for jj, j in enumerate(range(lo, hi + 1)):
            cur[j] = d[jj] + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m])


def dtw_1nn_scores(X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    Xd = X[:, ::DTW_DOWNSAMPLE, :]
    Xd = np.stack([_zscore_cols(x) for x in Xd])
    refs = Xd[train_idx]
    out = np.empty(len(Xd))
    for i, x in enumerate(Xd):
        best = np.inf
        for j, r in enumerate(refs):
            if train_idx[j] == i:
                continue
            best = min(best, _dtw_band(x, r))
        out[i] = best
        if (i + 1) % 50 == 0:
            print(f"    dtw {i + 1}/{len(Xd)}")
    return out


def engineered_features(X_raw: np.ndarray) -> np.ndarray:
    rr = X_raw[:, :, 0]
    rs = X_raw[:, :, 1]
    pa = X_raw[:, :, 2]

    sign_changes = np.sum(np.diff(np.sign(rr), axis=1) != 0, axis=1)
    swing = rr_swing_scores(rr)

    spec = np.abs(np.fft.rfft(rr - rr.mean(axis=1, keepdims=True), axis=1))
    peak_power = spec[:, 1:].max(axis=1) / (spec[:, 1:].sum(axis=1) + 1e-9)

    feats = np.stack([
        np.abs(rr).max(axis=1),
        rr.std(axis=1),
        sign_changes.astype(float),
        swing,
        rs.min(axis=1),
        rs.max(axis=1),
        rs.max(axis=1) - rs.min(axis=1),
        rs.std(axis=1),
        pa.std(axis=1),
        pa.max(axis=1) - pa.min(axis=1),
        peak_power,
    ], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def ocsvm_scores(X_raw: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    F = engineered_features(X_raw)
    scaler = StandardScaler().fit(F[train_idx])
    Fz = scaler.transform(F)
    clf = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale").fit(Fz[train_idx])
    return -clf.decision_function(Fz)


def f1_best(y, s):
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    return float(np.max(f))


def p_at_1pct_fpr(y, s):
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    thr = np.quantile(neg, 0.99)
    return float(((s >= thr) & (y == 1)).sum() / max((s >= thr).sum(), 1))


def evaluate(name, scores_test, y_test, src_test, rng):
    strat = stratified_metrics(y_test, scores_test)

    real_pos = (y_test == 2)
    op_neg = np.isin(src_test, OPERATIONAL_NEG_SOURCES)
    s_pos, s_neg = scores_test[real_pos], scores_test[op_neg]
    n_pos, n_neg = len(s_pos), len(s_neg)
    y_op = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(int)
    s_op = np.r_[s_pos, s_neg]

    point_gap = float(s_pos.min() - s_neg.max())
    n_gap_pos = 0
    for _ in range(N_BOOT):
        sp = s_pos[rng.integers(0, n_pos, n_pos)]
        sn = s_neg[rng.integers(0, n_neg, n_neg)]
        if sp.min() - sn.max() > 0:
            n_gap_pos += 1

    return {
        "method": name,
        "test_all_auroc": strat["all"]["auroc"],
        "test_rpo_real_auroc": strat.get("rpo_real_only", {}).get("auroc", float("nan")),
        "operational_auroc": float(roc_auc_score(y_op, s_op)),
        "operational_f1": f1_best(y_op, s_op),
        "operational_p_at_1pct_fpr": p_at_1pct_fpr(y_op, s_op),
        "separation_gap": point_gap,
        "frac_gap_positive": n_gap_pos / N_BOOT,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=REPO_ROOT / "runs" / "physics_baselines.json")
    ap.add_argument("--skip-dtw", action="store_true")
    args = ap.parse_args()

    meta = pd.read_parquet(BENCH_DIR / "windows_meta.parquet") \
             .sort_values("window_id").reset_index(drop=True)
    feat = pd.read_parquet(BENCH_DIR / "windows_feat.parquet")

    X, y_all = to_tensor_per_window(meta, feat)
    X_rate = X[:, :, 1:]
    train_mask = (meta["split"] == "train").values
    test_mask = (meta["split"] == "test").values
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"windows {X.shape}, train {train_mask.sum()}, test {test_mask.sum()}")

    print("scoring cw_template ...")
    s_cw = cw_template_scores(X_rate)
    print("scoring rr_swing ...")
    s_rr = rr_swing_scores(X_rate[:, :, 0])
    print("scoring ocsvm_feat ...")
    s_oc = ocsvm_scores(X_rate, train_idx)
    scores = {"cw_template": s_cw, "rr_swing": s_rr, "ocsvm_feat": s_oc}

    if not args.skip_dtw:
        print("scoring dtw_1nn ...")
        scores["dtw_1nn"] = dtw_1nn_scores(X_rate, train_idx)

    src = meta["source"].str.split(":").str[0].values
    y_test, src_test = y_all[test_idx], src[test_idx]
    rng = np.random.default_rng(0)

    rows = [evaluate(n, s[test_idx], y_test, src_test, rng)
            for n, s in scores.items()]

    hdr = (f"{'method':14s} {'test all':>9s} {'rpo_real':>9s} "
           f"{'op AUROC':>9s} {'op F1':>7s} {'P@1%':>7s} {'gap':>9s} {'boot':>7s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:14s} {r['test_all_auroc']:>9.3f} "
              f"{r['test_rpo_real_auroc']:>9.3f} {r['operational_auroc']:>9.3f} "
              f"{r['operational_f1']:>7.3f} {r['operational_p_at_1pct_fpr']:>7.3f} "
              f"{r['separation_gap']:>+9.4f} {r['frac_gap_positive']:>7.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
