"""Generate the committed cross-language parity fixtures.

Usage (from the project directory, so engine/ imports and the default
BgMoveGen DLL path resolve):

    python -m parity.generate_vectors

Produces, next to this script:

    model.onnx    tiny deterministic parity model (committed)
    vectors.json  golden (board -> features -> output) triples (committed)

These two files are the executable contract the C# consumer (BgInference)
must reproduce: the feature encoding cannot be single-sourced between
Python and C#, so the vectors are the cross-language SSOT mitigation.
Both are committed — a fresh checkout must fail the C# parity gate loudly
if they are wrong, never no-op because they are absent.

The parity model is NOT a trained model and is deliberately tiny
(303 -> 16 -> 16 -> 6, ~25 KB): the gate proves encoding and inference
parity, not playing strength, and a width-16 net exercises the identical
op set (Gemm / Relu / Sigmoid), dynamic-batch graph contract, and
metadata handshake as any production export. Its weights come from a
seeded NumPy PRNG (not torch init, whose streams are not guaranteed
stable across versions), so the committed binary is regenerable and
auditable. It changes only when the contract changes (encoding version,
graph shape, opset) — trained models are deployment artifacts, exported
locally via export_onnx.py, and are never committed.

Vector outputs are produced by running model.onnx under ONNX Runtime —
the same artifact and runtime stack the consumer uses — not by the
PyTorch network (tests/test_parity.py separately pins torch vs ORT).

Float serialization: features and outputs are float32 written as plain
JSON numbers. float32 -> float64 is exact and repr round-trips, so a
consumer parsing to double and casting to float recovers the exact bits;
the encoding pin is therefore bit-exact (feature_tolerance_abs = 0.0).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from engine.export import (
    INPUT_NAME,
    OUTPUT_NAME,
    OUTPUT_SEMANTICS,
    export_network,
)
from engine.movegen import (
    REQUIRED_MOVEGEN_VERSION,
    Variant,
    generate_successor_states,
    get_starting_position,
    load_movegen,
)
from engine.network import NUM_OUTPUTS, TDNetwork
from engine.state import (
    BOARD_FEATURE_SIZE,
    ENCODING_VERSION,
    NUM_POINTS,
    BoardState,
    encode_board,
    encode_board_batch,
    flip_perspective,
)

PARITY_DIR = Path(__file__).parent
MODEL_PATH = PARITY_DIR / "model.onnx"
VECTORS_PATH = PARITY_DIR / "vectors.json"

PARITY_HIDDEN_LAYERS = [16, 16]
WEIGHT_SEED = 20260702
PLAYOUT_SEED = 7
BG960_START_SEEDS = (1, 2, 3)
MIDGAME_SNAPSHOT_PLIES = (8, 16, 24)

# Consumer-gate tolerances, carried inside the fixture so the C# gate
# reads them from here rather than hardcoding its own copy.
FEATURE_TOLERANCE_ABS = 0.0  # bit-exact
OUTPUT_TOLERANCE_ABS = 1e-5


def build_parity_network(seed: int) -> TDNetwork:
    """Build the tiny parity network with seeded-NumPy deterministic weights.

    Xavier-uniform-style limits are computed manually in NumPy so the
    weights do not depend on torch's initialization streams.
    """
    rng = np.random.default_rng(seed)
    network = TDNetwork(
        input_size=BOARD_FEATURE_SIZE, hidden_layers=PARITY_HIDDEN_LAYERS
    )
    state_dict = {}
    for key, tensor in network.state_dict().items():
        shape = tuple(tensor.shape)
        if key.endswith(".weight"):
            fan_out, fan_in = shape
            limit = math.sqrt(6.0 / (fan_in + fan_out))
        else:
            limit = 0.05  # non-zero biases exercise the Gemm add
        values = rng.uniform(-limit, limit, size=shape).astype(np.float32)
        state_dict[key] = torch.from_numpy(values)
    network.load_state_dict(state_dict)
    return network.eval()


def _board(
    points: dict[int, int],
    *,
    bar_player: int = 0,
    bar_opponent: int = 0,
    off_player: int = 0,
    off_opponent: int = 0,
    player_to_move: bool = True,
) -> BoardState:
    """Hand-craft a BoardState from a sparse {point index: signed count} map."""
    setup = np.zeros(NUM_POINTS, dtype=np.int16)
    for index, count in points.items():
        setup[index] = count
    state = BoardState.from_setup(setup)
    state.bar_player = bar_player
    state.bar_opponent = bar_opponent
    state.off_player = off_player
    state.off_opponent = off_opponent
    state.player_to_move = player_to_move
    return state


def _with_opponent_to_move(state: BoardState) -> BoardState:
    copy = state.copy()
    copy.player_to_move = False
    return copy


def _midgame_positions(
    variant: Variant, rng: np.random.Generator
) -> list[tuple[int, BoardState]]:
    """Seeded random playout; snapshot positions at fixed plies."""
    if variant == Variant.BG960:
        state = get_starting_position(variant, seed=PLAYOUT_SEED)
    else:
        state = get_starting_position(variant)
    snapshots = []
    for ply in range(1, max(MIDGAME_SNAPSHOT_PLIES) + 1):
        die1 = int(rng.integers(1, 7))
        die2 = int(rng.integers(1, 7))
        successors = generate_successor_states(state, die1, die2)
        state = successors[int(rng.integers(len(successors)))]
        if ply in MIDGAME_SNAPSHOT_PLIES:
            snapshots.append((ply, state.copy()))
    return snapshots


def build_cases() -> list[tuple[str, BoardState]]:
    """The fixture case set: starts, midgames, and degenerate boards."""
    cases: list[tuple[str, BoardState]] = []

    # Starting positions — all variants; Bg960 across several seeds.
    standard_start = get_starting_position(Variant.STANDARD)
    cases.append(("standard-start", standard_start))
    cases.append(
        ("standard-start-opponent-to-move",
         _with_opponent_to_move(standard_start))
    )
    cases.append(("nackgammon-start", get_starting_position(Variant.NACKGAMMON)))
    for seed in BG960_START_SEEDS:
        cases.append(
            (f"bg960-start-seed{seed}",
             get_starting_position(Variant.BG960, seed=seed))
        )

    # Midgame positions — seeded playouts per variant.
    rng = np.random.default_rng(PLAYOUT_SEED)
    for variant in (Variant.STANDARD, Variant.NACKGAMMON, Variant.BG960):
        for ply, state in _midgame_positions(variant, rng):
            cases.append((f"{variant.name.lower()}-midgame-ply{ply}", state))

    # Perspective flip and side-to-move variation on a midgame position.
    standard_midgame = next(
        state for label, state in cases if label == "standard-midgame-ply16"
    )
    cases.append(
        ("standard-midgame-ply16-flipped", flip_perspective(standard_midgame))
    )
    cases.append(
        ("standard-midgame-ply16-opponent-to-move",
         _with_opponent_to_move(standard_midgame))
    )

    # Pure races (no contact possible).
    cases.append((
        "race-symmetric",
        _board({0: 3, 1: 3, 2: 3, 3: 3, 4: 3,
                19: -3, 20: -3, 21: -3, 22: -3, 23: -3}),
    ))
    cases.append((
        "race-asymmetric",
        _board({0: 2, 2: 4, 5: 5, 7: 3, 8: 1,
                15: -1, 17: -4, 20: -6, 22: -2, 23: -2}),
    ))

    # Bear-off.
    cases.append((
        "bearoff-both",
        _board({0: 2, 1: 2, 2: 1, 21: -3, 22: -2, 23: -2},
               off_player=10, off_opponent=8),
    ))
    cases.append((
        "bearoff-player-last-checker",
        _board({0: 1, 18: -2, 20: -3}, off_player=14, off_opponent=10),
    ))

    # Game-over shapes.
    cases.append((
        "gameover-player-won",
        _board({20: -4, 21: -3}, off_player=15, off_opponent=8),
    ))
    cases.append((
        "gameover-gammon",
        _board({5: -5, 11: -4, 18: -6}, off_player=15, off_opponent=0),
    ))

    # Degenerate: empty board, everything borne off — exercises the
    # total_pips == 0 branch of the pip-ratio feature.
    cases.append((
        "degenerate-empty-all-off",
        _board({}, off_player=15, off_opponent=15),
    ))

    # Bar-heavy.
    cases.append((
        "bar-heavy",
        _board({5: 4, 7: 4, 12: 4, 16: -3, 18: -4, 20: -4},
               bar_player=3, bar_opponent=4),
    ))
    cases.append((
        "bar-one-each",
        _board({5: 5, 7: 3, 12: 5, 0: -1, 11: -5, 16: -3, 18: -5},
               bar_player=1, bar_opponent=1, player_to_move=False),
    ))

    # Thermometer overflow (> 5 checkers on a point).
    cases.append((
        "overflow-single-point",
        _board({5: 15, 18: -15}),
    ))
    cases.append((
        "overflow-split",
        _board({3: 7, 9: 8, 14: -8, 21: -7}),
    ))

    return cases


def _serialize_case(
    label: str, state: BoardState, features: np.ndarray, output: np.ndarray
) -> dict:
    return {
        "label": label,
        "board": {
            "points": [int(p) for p in state.points],
            "bar_player": state.bar_player,
            "bar_opponent": state.bar_opponent,
            "off_player": state.off_player,
            "off_opponent": state.off_opponent,
            "player_to_move": state.player_to_move,
        },
        "features": [float(f) for f in features],
        "output": [float(o) for o in output],
    }


def _dump_compact_arrays(document: dict) -> str:
    """Pretty-print JSON, but keep scalar arrays on a single line."""
    text = json.dumps(document, indent=2)
    return re.sub(
        r"\[\s+([^\[\]{}]+?)\s+\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
        text,
        flags=re.S,
    )


def main() -> None:
    load_movegen()

    # 1. Deterministic parity model -> committed ONNX.
    network = build_parity_network(WEIGHT_SEED)
    export_network(
        network,
        MODEL_PATH,
        model_role="parity",
        provenance={
            "weight_seed": str(WEIGHT_SEED),
            "generator": "parity/generate_vectors.py",
        },
    )
    model_sha256 = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    print(f"Wrote {MODEL_PATH} (sha256 {model_sha256[:12]}...)")

    # 2. Case set -> features (scalar reference path, cross-checked
    #    against the vectorized path) -> outputs via ONNX Runtime.
    cases = build_cases()
    labels = [label for label, _ in cases]
    assert len(set(labels)) == len(labels), "duplicate case labels"

    features = np.stack([encode_board(state) for _, state in cases])
    batch_features = encode_board_batch([state for _, state in cases])
    assert np.array_equal(features, batch_features), (
        "encode_board and encode_board_batch disagree — fix that before "
        "blessing parity vectors"
    )

    session = ort.InferenceSession(
        str(MODEL_PATH), providers=["CPUExecutionProvider"]
    )
    outputs = session.run([OUTPUT_NAME], {INPUT_NAME: features})[0]
    assert outputs.shape == (len(cases), NUM_OUTPUTS)

    # Sanity: ORT vs the torch network it was exported from.
    with torch.no_grad():
        torch_outputs = network(torch.from_numpy(features)).numpy()
    assert np.allclose(outputs, torch_outputs, rtol=0, atol=1e-6)

    document = {
        "format_version": 1,
        "encoding_version": ENCODING_VERSION,
        "input_size": BOARD_FEATURE_SIZE,
        "num_outputs": NUM_OUTPUTS,
        "output_semantics": OUTPUT_SEMANTICS,
        "model_file": MODEL_PATH.name,
        "model_sha256": model_sha256,
        "feature_tolerance_abs": FEATURE_TOLERANCE_ABS,
        "output_tolerance_abs": OUTPUT_TOLERANCE_ABS,
        "generator": {
            "script": "parity/generate_vectors.py",
            "weight_seed": WEIGHT_SEED,
            "playout_seed": PLAYOUT_SEED,
            "movegen_version": REQUIRED_MOVEGEN_VERSION,
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
        },
        "cases": [
            _serialize_case(label, state, features[i], outputs[i])
            for i, (label, state) in enumerate(cases)
        ],
    }

    VECTORS_PATH.write_text(
        _dump_compact_arrays(document) + "\n", encoding="utf-8"
    )
    print(f"Wrote {VECTORS_PATH} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
