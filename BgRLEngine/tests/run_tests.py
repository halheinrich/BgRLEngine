"""Standalone tests — runs without pytest or torch.

Tests the pure-Python/numpy modules: state encoding, setup generator,
SPRT logic, and dice/move generation.

Usage (from BgRLEngine\BgRLEngine\):
    python tests/run_tests.py
"""

import sys
import traceback
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            errors.append((name, traceback.format_exc()))
            print(f"  ✗ {name}: {e}")
        return fn
    return decorator


# ── Load BgMoveGen ─────────────────────────────────────────────────

import yaml
from engine.movegen import load_movegen, get_starting_position, Variant

config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
with open(config_path, encoding="utf-8") as f:
    _config = yaml.safe_load(f)
load_movegen(_config["movegen"]["dll_path"])


# ── State encoding ─────────────────────────────────────────────────

from engine.state import (
    BoardState, encode_board, encode_point, encode_bar,
    encode_borne_off, flip_perspective,
    BOARD_FEATURE_SIZE, NUM_POINTS, UNITS_PER_POINT,
)

print("\n── State encoding ──")


@test("encode_point: zero checkers")
def _():
    f = encode_point(0)
    assert len(f) == UNITS_PER_POINT
    assert all(f == 0)


@test("encode_point: 1 checker")
def _():
    f = encode_point(1)
    assert f[0] == 1.0 and all(f[1:] == 0)


@test("encode_point: 5 checkers (thermometer full)")
def _():
    f = encode_point(5)
    assert all(f[:5] == 1.0) and f[5] == 0.0


@test("encode_point: 8 checkers (overflow)")
def _():
    f = encode_point(8)
    assert all(f[:5] == 1.0)
    assert abs(f[5] - 0.3) < 1e-6


@test("encode_point: 15 checkers (overflow capped)")
def _():
    f = encode_point(15)
    assert all(f[:5] == 1.0)
    assert f[5] == 1.0


@test("encode_bar: 0, 1, 5")
def _():
    assert all(encode_bar(0) == 0)
    f1 = encode_bar(1)
    assert f1[0] == 1.0 and f1[1] == 0.0
    assert all(encode_bar(5) == 1.0)


@test("encode_borne_off: none, some, all")
def _():
    f = encode_borne_off(0)
    assert f[0] == 0.0 and f[1] == 0.0
    f = encode_borne_off(7)
    assert abs(f[0] - 7/15) < 1e-6 and f[1] == 0.0
    f = encode_borne_off(15)
    assert abs(f[0] - 1.0) < 1e-6 and f[1] == 1.0


@test("standard setup: 15 checkers per side")
def _():
    state = BoardState.standard_setup()
    player = sum(max(0, state.points[i]) for i in range(24))
    opponent = sum(abs(min(0, state.points[i])) for i in range(24))
    assert player == 15, f"Player has {player}"
    assert opponent == 15, f"Opponent has {opponent}"


@test("standard setup: pip count = 167")
def _():
    state = BoardState.standard_setup()
    assert state.player_pip_count() == 167, f"Got {state.player_pip_count()}"
    assert state.opponent_pip_count() == 167, f"Got {state.opponent_pip_count()}"


@test("nackgammon: 15 checkers per side (via BgMoveGen)")
def _():
    state = get_starting_position(Variant.NACKGAMMON)
    player = sum(max(0, state.points[i]) for i in range(24))
    opponent = sum(abs(min(0, state.points[i])) for i in range(24))
    assert player == 15, f"Player has {player}"
    assert opponent == 15, f"Opponent has {opponent}"


@test("nackgammon: authoritative layout (via BgMoveGen)")
def _():
    state = get_starting_position(Variant.NACKGAMMON)
    assert state.points[5]  ==  4
    assert state.points[7]  ==  3
    assert state.points[12] ==  4
    assert state.points[22] ==  2
    assert state.points[23] ==  2
    assert state.points[18] == -4
    assert state.points[16] == -3
    assert state.points[11] == -4
    assert state.points[1]  == -2
    assert state.points[0]  == -2


@test("standard setup: not a race")
def _():
    assert not BoardState.standard_setup().is_race()


@test("separated position: is a race")
def _():
    state = BoardState()
    state.points[0] = 5; state.points[3] = 5; state.points[5] = 5
    state.points[18] = -5; state.points[20] = -5; state.points[23] = -5
    assert state.is_race()


@test("encode_board: correct size and dtype")
def _():
    features = encode_board(BoardState.standard_setup())
    assert len(features) == BOARD_FEATURE_SIZE, f"Got {len(features)}"
    assert features.dtype == np.float32


@test("flip_perspective: preserves checker counts")
def _():
    state = BoardState.standard_setup()
    flipped = flip_perspective(state)
    p_orig = sum(max(0, state.points[i]) for i in range(24))
    p_flip = sum(max(0, flipped.points[i]) for i in range(24))
    o_orig = sum(abs(min(0, state.points[i])) for i in range(24))
    o_flip = sum(abs(min(0, flipped.points[i])) for i in range(24))
    assert p_flip == o_orig
    assert o_flip == p_orig


@test("double flip = identity")
def _():
    state = BoardState.standard_setup()
    double = flip_perspective(flip_perspective(state))
    np.testing.assert_array_equal(state.points, double.points)
    assert state.bar_player == double.bar_player
    assert state.bar_opponent == double.bar_opponent


# ── Setup generator ────────────────────────────────────────────────

from engine.setup_generator import SetupGenerator, QUADRANTS

print("\n── Setup generator ──")


@test("generates 15 player checkers")
def _():
    gen = SetupGenerator(rng=np.random.default_rng(42))
    for _ in range(20):
        state = gen.generate()
        count = sum(max(0, state.points[i]) for i in range(24))
        assert count == 15, f"Got {count}"


@test("symmetry: player mirrors opponent")
def _():
    gen = SetupGenerator(rng=np.random.default_rng(42))
    for _ in range(20):
        state = gen.generate()
        for i in range(24):
            mirror = 23 - i
            player_here = max(0, state.points[i])
            opp_mirror = abs(min(0, state.points[mirror]))
            assert player_here == opp_mirror, (
                f"Point {i}: player={player_here}, opp@mirror={opp_mirror}"
            )


@test("min 2 checkers per occupied point")
def _():
    gen = SetupGenerator(rng=np.random.default_rng(42))
    for _ in range(50):
        state = gen.generate()
        for i in range(24):
            if state.points[i] > 0:
                assert state.points[i] >= 2, f"Point {i} has {state.points[i]}"


@test("at least one point per quadrant")
def _():
    gen = SetupGenerator(rng=np.random.default_rng(42))
    for _ in range(50):
        state = gen.generate()
        for q in QUADRANTS:
            has = any(state.points[i] > 0 for i in q)
            assert has, f"Quadrant {list(q)} empty"


@test("pip count >= 100")
def _():
    gen = SetupGenerator(rng=np.random.default_rng(42))
    for _ in range(50):
        state = gen.generate()
        pips = state.player_pip_count()
        assert pips >= 100, f"Pips = {pips}"


@test("batch generation")
def _():
    gen = SetupGenerator(rng=np.random.default_rng(42))
    batch = gen.generate_batch(10)
    assert len(batch) == 10


@test("weight update doesn't break generation")
def _():
    gen = SetupGenerator(rng=np.random.default_rng(42))
    gen.set_weights({4: 1, 5: 1, 6: 1, 7: 1})
    state = gen.generate()
    assert sum(max(0, state.points[i]) for i in range(24)) == 15


# ── Dice and moves ─────────────────────────────────────────────────

from engine.dice import roll_dice, get_dice_to_use, generate_plays

print("\n── Dice and moves ──")


@test("roll_dice: values 1-6")
def _():
    rng = np.random.default_rng(42)
    for _ in range(200):
        d1, d2 = roll_dice(rng)
        assert 1 <= d1 <= 6 and 1 <= d2 <= 6


@test("doubles give 4 dice, non-doubles give 2")
def _():
    assert len(get_dice_to_use(3, 3)) == 4
    assert len(get_dice_to_use(3, 5)) == 2


@test("opening position has legal plays")
def _():
    state = BoardState.standard_setup()
    plays = generate_plays(state, 3, 1)
    assert len(plays) > 0


@test("blocked bar = no legal moves")
def _():
    state = BoardState()
    state.bar_player = 1
    for i in range(18, 24):
        state.points[i] = -2
    plays = generate_plays(state, 3, 1)
    assert len(plays) == 1 and plays[0].num_moves == 0


@test("bearing off works")
def _():
    state = BoardState()
    state.points[5] = 5; state.points[3] = 5; state.points[1] = 5
    state.points[23] = -15
    plays = generate_plays(state, 6, 4)
    assert len(plays) > 0
    has_bear_off = any(
        any(m.dest == -1 for m in p.moves) for p in plays
    )
    assert has_bear_off


@test("must use both dice if possible")
def _():
    state = BoardState.standard_setup()
    plays = generate_plays(state, 6, 1)
    for p in plays:
        assert p.num_moves == 2, f"Play uses {p.num_moves} moves"


# ── SPRT ───────────────────────────────────────────────────────────

from utils.sprt import sprt_test, SPRTResult

print("\n── SPRT ──")


@test("strong wins → accept")
def _():
    assert sprt_test(95, 100) == SPRTResult.ACCEPT


@test("strong losses → reject")
def _():
    assert sprt_test(30, 100) == SPRTResult.REJECT


@test("ambiguous → continue")
def _():
    assert sprt_test(73, 100) == SPRTResult.CONTINUE


@test("hard cap → reject")
def _():
    assert sprt_test(1460, 2000) == SPRTResult.REJECT


@test("zero games → continue")
def _():
    assert sprt_test(0, 0) == SPRTResult.CONTINUE


# ── Summary ────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print(f"\nFailed tests:")
    for name, tb in errors:
        print(f"\n  {name}:")
        for line in tb.strip().split("\n"):
            print(f"    {line}")

sys.exit(1 if failed else 0)