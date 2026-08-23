from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    print("=" * 70)
    print("Table 2: Per-source AUROC breakdown")
    print("=" * 70)
    print("\nTrains 4 methods x 5 seeds, score-averages, then computes")
    print("binary AUROC for each (pos, neg) source pair.")
    print("Takes ~10-15 min on a single CPU core.\n")

    from pairwatch.eval.per_source_breakdown import main as ps_main
    ps_main()

    print("\nSame decomposition over the non-neural detectors...")
    from pairwatch.eval.physics_per_source import main as pps_main
    pps_main()

    print("\n" + "=" * 70)
    print(" Outputs: runs/per_source.json, runs/physics_per_source.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
