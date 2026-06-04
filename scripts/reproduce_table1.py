from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    print("=" * 70)
    print("Table 1: Main results on the rate-only benchmark")
    print("=" * 70)

    print("\n[1/3] Single-satellite Isolation Forest baseline...")
    from pairwatch.eval.iforest import main as iforest_main
    from pairwatch.eval.iforest_operational import main as iforest_op_main
    iforest_main()
    iforest_op_main()

    print("\n[2/3] Neural detectors (5 seeds, score-averaging)...")
    print("      LSTM-AD, TranAD, Vanilla AT, PairWatch (Fix A)")
    print("      This takes ~10-15 min on a single CPU core.")
    from pairwatch.eval.significance_fixA import main as sig_main
    sig_main()

    print("\n[3/3] Operational subset metrics + bootstrap robustness...")
    from pairwatch.eval.operational_and_regime import main as opreg_main
    opreg_main()
    from pairwatch.eval.operational_bootstrap import main as opboot_main
    opboot_main()

    print("\n" + "=" * 70)
    print(" Table 1 outputs:")
    print(f"   runs/iforest_current.json       (IF row)")
    print(f"   runs/iforest_operational.json   (IF operational row)")
    print(f"   runs/significance_fixA.json     (neural AUROC + DeLong)")
    print(f"   runs/operational_and_regime.json (operational metrics)")
    print(f"   runs/operational_bootstrap.json (separation bootstrap)")
    print("=" * 70)


if __name__ == "__main__":
    main()
