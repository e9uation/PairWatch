from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    print("=" * 70)
    print("Table 4: Parameters and runtime")
    print("=" * 70)
    print("\nTakes ~5-10 min on a single CPU core.\n")

    from pairwatch.eval.compute_cost import main as cc_main
    cc_main()

    print("\n" + "=" * 70)
    print(" Output: runs/compute_cost.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
