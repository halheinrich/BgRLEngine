"""
verify_bg960.py — Verify BgMoveGen Bg960 starting position constraints.

Generates N positions and checks all constraints:
  - 15 player checkers, 15 opponent checkers
  - Symmetrical (opponent mirrors player)
  - No blots (>= 2 checkers per occupied point)
  - At least one occupied point per quadrant
  - No mirror conflicts (point i and point 23-i never both occupied)
  - Minimum pip count >= 100
  - No checkers on bar or borne off
  - Seed reproducibility

Usage (from BgRLEngine\BgRLEngine\):
    python tests/verify_bg960.py
    python tests/verify_bg960.py --n 500
"""

import sys
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200, help="Number of positions to check")
args = parser.parse_args()

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from engine.movegen import load_movegen, get_starting_position, Variant
from engine.state import BoardState, NUM_POINTS

config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)
load_movegen(config["movegen"]["dll_path"])

QUADRANTS = [range(0, 6), range(6, 12), range(12, 18), range(18, 24)]

passed = 0
failed = 0
errors = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        errors.append(f"{name}: {detail}")

print(f"\nVerifying {args.n} Bg960 positions from BgMoveGen...\n")

for i in range(args.n):
    s = get_starting_position(Variant.BG960)

    player_counts   = np.maximum(s.points,  0)
    opponent_counts = np.maximum(-s.points, 0)

    # 1. Checker counts
    check(f"[{i}] player=15",
          player_counts.sum() == 15,
          f"got {player_counts.sum()}")

    check(f"[{i}] opponent=15",
          opponent_counts.sum() == 15,
          f"got {opponent_counts.sum()}")

    # 2. No bar or borne off
    check(f"[{i}] bar=0",
          s.bar_player == 0 and s.bar_opponent == 0,
          f"bar_player={s.bar_player} bar_opponent={s.bar_opponent}")

    check(f"[{i}] off=0",
          s.off_player == 0 and s.off_opponent == 0,
          f"off_player={s.off_player} off_opponent={s.off_opponent}")

    # 3. Symmetry: player on point i ↔ opponent on point 23-i
    sym_ok = all(
        player_counts[i] == opponent_counts[NUM_POINTS - 1 - i]
        for i in range(NUM_POINTS)
    )
    check(f"[{i}] symmetry", sym_ok,
          f"points={s.points.tolist()}")

    # 4. No blots (each occupied point has >= 2 checkers)
    no_blots = all(
        s.points[j] == 0 or abs(s.points[j]) >= 2
        for j in range(NUM_POINTS)
    )
    check(f"[{i}] no blots", no_blots,
          f"points={s.points.tolist()}")

    # 5. At least one player point per quadrant
    for q in QUADRANTS:
        check(f"[{i}] quadrant {list(q)[0]}-{list(q)[-1]}",
              any(s.points[j] > 0 for j in q),
              f"no player checker in quadrant {list(q)}")

    # 6. No mirror conflicts
    no_mirror = all(
        not (s.points[j] > 0 and s.points[NUM_POINTS - 1 - j] > 0)
        for j in range(NUM_POINTS)
    )
    check(f"[{i}] no mirror conflicts", no_mirror,
          f"points={s.points.tolist()}")

    # 7. Pip count >= 100
    p_pips = sum(int(player_counts[j]) * (j + 1) for j in range(NUM_POINTS))
    check(f"[{i}] pip count >= 100", p_pips >= 100,
          f"pip count = {p_pips}")

# 8. Seed reproducibility — same seed should give same position
s1 = get_starting_position(Variant.BG960, seed=42)
s2 = get_starting_position(Variant.BG960, seed=42)
check("seed reproducibility",
      np.array_equal(s1.points, s2.points),
      f"s1={s1.points.tolist()} s2={s2.points.tolist()}")

# 9. Different seeds give different positions (with high probability)
s3 = get_starting_position(Variant.BG960, seed=99)
check("different seeds → different positions",
      not np.array_equal(s1.points, s3.points),
      "seed=42 and seed=99 gave identical positions")

# ── Report ──────────────────────────────────────────────────────────
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print(f"\nFailed checks ({len(errors)}):")
    for e in errors[:20]:  # cap output
        print(f"  ✗ {e}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")
else:
    print("  ✓ All constraints satisfied across all positions")
print()

sys.exit(1 if failed else 0)