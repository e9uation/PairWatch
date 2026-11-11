# PairWatch

[![arXiv](https://img.shields.io/badge/arXiv-2610.39679-b31b1b.svg)](https://arxiv.org/abs/2610.39679)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3799682.3839977-blue.svg)](https://doi.org/10.1145/3799682.3839977)
[![Code License](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/data-CC--BY--4.0-orange.svg)](benchmark/DATA_LICENSE)

Pair-aware anomaly detection for adversarial rendezvous and proximity operations (RPO) on public TLE catalogs.

This repository accompanies the CIKM 2026 short paper [*"PairWatch: Pair-Aware Anomaly Transformer for Detecting Adversarial Rendezvous in Public Satellite Catalogs"*](https://arxiv.org/abs/2610.39679) ([ACM DL](https://doi.org/10.1145/3799682.3839977)).

It contains the open benchmark of 359 pair-windows, the PairWatch implementation, all nine baseline detectors, four reproduction scripts, and the data collection pipeline.

## Repository Structure

```
PairWatch/
├── benchmark/                   Open RPO benchmark (processed features)
│   ├── windows_meta.parquet     Window metadata (target/neighbor IDs, timestamps, label, source, split)
│   ├── windows_feat.parquet     Pair feature vectors at hourly resolution over a 14-day window (range, range rate, relative speed, plane-alignment angle)
│   └── DATA_LICENSE             CC-BY 4.0
├── pairwatch/                   PairWatch implementation
│   ├── eval/                    All baselines + PairWatch + metrics
│   ├── synth/                   Clohessy-Wiltshire generator
│   └── data/                    Benchmark construction pipeline
├── scripts/                     One script per paper table
│   ├── reproduce_table1.py      Main results (5-seed AUROC + bootstrap)
│   ├── reproduce_table2.py      Per-source breakdown
│   ├── reproduce_table3.py      Ablation
│   └── reproduce_table4.py      Compute cost
├── data_collection/             Raw TLE collection (Space-Track)
│   └── fetch_tle.py             For users who want to rebuild from scratch
├── requirements.txt
├── LICENSE                      MIT (code)
└── README.md
```

## Baseline detectors

Nine detectors are evaluated on the benchmark. Eight score the same rate-only pair-feature input; Isolation Forest instead scores mean-element residuals of the target alone, which is what makes it the single-satellite control.

| Detector | Family | Fitted on | Module |
|----------|--------|-----------|--------|
| Isolation Forest (single-sat) | classical | train split | `eval/iforest.py` |
| Range-rate swing | physics heuristic | — | `eval/physics_baselines.py` |
| CW template matching | physics model | — | `eval/physics_baselines.py` |
| DTW 1-NN | elastic distance | train split | `eval/physics_baselines.py` |
| One-class SVM (pair feats) | classical | train split | `eval/physics_baselines.py` |
| LSTM-AD | neural | train split | `eval/lstm_ad.py` |
| TranAD | neural | train split | `eval/tran_ad.py` |
| Vanilla AnomalyTransformer | neural | train split | `eval/anomaly_transformer.py` |
| PairWatch | neural | train split | `eval/pairwatch.py` |

The range-rate, template, DTW, and one-class detectors are deterministic and need no seed averaging. Every fitted detector sees only the train split, which contains negatives alone; Isolation Forest is fitted with a fixed seed, and the neural detectors are run over five seeds with their per-window scores averaged.

## Run & Reproduce

**Requirements**: Python 3.9+ (tested on 3.11)

### 1. Create environment
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce paper tables
```bash
# Main results (Table 1)
python scripts/reproduce_table1.py

# Per-source breakdown (Table 2)
python scripts/reproduce_table2.py

# Ablation study (Table 3)
python scripts/reproduce_table3.py

# Compute cost evaluation (Table 4)
python scripts/reproduce_table4.py
```

> **Note — the Isolation Forest row (Table 1, first row).** This one baseline reads raw TLE from data/raw_tle/, which we cannot redistribute under Space-Track's terms of service. Without it, every window falls back to a zero feature vector and the row reports a degenerate AUROC of **0.500** in place of the published 0.600 / 0.228 / 0.175. This is a packaging limitation, not a discrepancy in the reported results: the row reproduces exactly once raw TLE is present (see [Rebuilding the benchmark from raw TLE](#rebuilding-the-benchmark-from-raw-tle-optional)). Every other row of Table 1 and all of Tables 2–4 reproduce from the bundled benchmark alone, with no Space-Track account required.

All outputs, logs, and generated tables will be saved in `runs/` directory.

## Benchmark contents

The benchmark is a single train/val/test split over 359 pair-windows. Each window spans 14 days at an hourly resolution (336 timesteps per window).

### Dataset summary

All 11 real-RPO windows are forced into the test split to ensure rigorous evaluation.

| Source                 | Train | Val | Test | Total |
|------------------------|------:|----:|-----:|------:|
| rpo_real               |     0 |   0 |   11 |    11 |
| rpo_synth              |     0 |  45 |   55 |   100 |
| clean                  |    46 |  20 |   24 |    90 |
| routine_maneuver       |    15 |  17 |   22 |    54 |
| starlink_constellation |     0 |  20 |   36 |    56 |
| formation_flying       |     0 |  10 |   38 |    48 |
| **Total**              |    61 | 112 |  186 |   359 |


### Window metadata columns

| Column         | Description                                        |
|----------------|----------------------------------------------------|
| window_id      | Unique identifier                                  |
| target_id      | NORAD ID of target satellite (or synthetic ref.)   |
| neighbor_id    | NORAD ID of neighbor / attacker                    |
| t_start        | Window start time (UTC, ISO 8601)                  |
| t_end          | Window end time (UTC, ISO 8601)                    |
| label          | 0/1/2/3 (negative classes / real_rpo / synth_rpo)  |
| source         | Categorical source label                           |
| split          | train / val / test                                 |

### Pair feature columns

Each row represents one hourly timestep within a specific window.

| Column            | Units  | Description                              |
|-------------------|--------|------------------------------------------|
| window_id         | -      | Foreign key to windows_meta              |
| t                 | hours  | Timestep index (0..335)                  |
| range_km          | km     | Inertial range \|r_A - r_T\|             |
| range_rate_km_s   | km/s   | d/dt of range                            |
| rel_speed_km_s    | km/s   | \|v_A - v_T\|                            |
| plane_angle_deg   | deg    | Orbital-plane alignment angle            |

Note: paper uses *rate-only setting* (range_km withheld; only the three rate-of-change channels used for inference).

## Rebuilding the benchmark from raw TLE (optional)

The provided pre-processed features are sufficient for all replication steps. However, if you wish to extend the dataset or rebuild it from scratch (e.g., add new RPO events), you can use `data_collection/fetch_tle.py`.

### Space-Track Terms of Service

Raw TLE data is hosted at https://www.space-track.org under their terms of service, which prohibit redistribution of raw TLE files. To comply with these terms, we do not include raw TLEs in this repository.

To fetch the raw data independently:

1. Register a free account at Space-Track.
2. Set credentials in `.env` (see `.env.example`).
3. Execute the collection script:
   ```bash
   python -m data_collection.fetch_tle --ids [NORAD_IDs]
   ```

Our processed benchmark contains only derived kinematic features and anonymized metadata, making it fully eligible for open redistribution under the CC-BY 4.0 license.

## Citation

```bibtex
@inproceedings{kwon2026pairwatch,
  author        = {Kwon, Young Eun and Yoon, Ji Won},
  title         = {PairWatch: Pair-Aware Anomaly Transformer for Detecting Adversarial Rendezvous in Public Satellite Catalogs},
  booktitle     = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM '26)},
  year          = {2026},
  publisher     = {ACM},
  address       = {New York, NY, USA},
  doi           = {10.1145/3799682.3839977},
  eprint        = {2610.39679},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}
```

## License

- Code (`pairwatch/`, `scripts/`, `data_collection/`): MIT (see `LICENSE`).
- Benchmark data (`benchmark/`): CC-BY 4.0 (see `benchmark/DATA_LICENSE`).
- Raw TLE: NOT included; collect from Space-Track under their ToS.
