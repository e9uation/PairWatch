from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    print("=" * 70)
    print("Table 3: Ablation on the rate-only benchmark")
    print("=" * 70)
    print("\nTrains 7 PairWatch variants x 5 seeds = 35 training runs.")
    print("Takes ~30-40 min on a single CPU core.\n")

    from pairwatch.eval.ablation_table3 import main as abl_main
    abl_main()

    print("\n" + "=" * 70)
    print(" Output: runs/ablation_table3.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
