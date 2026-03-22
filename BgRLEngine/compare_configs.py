"""Find positions where the best play differs between equity weight configs.

Loads all four trained models, generates positions, and for each
position+dice roll finds where the models disagree on the best play.
Reports the top 3 equity differences for each of the 6 model pairings.

Usage:
    python compare_configs.py [--num-positions 500]
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path

from engine.state import BoardState, encode_board, flip_perspective, BOARD_FEATURE_SIZE
from engine.network import TDNetwork, compute_equity
from engine.dice import generate_plays, roll_dice, _apply_move
from engine.setup_generator import SetupGenerator
from engine.game import _apply_play


@dataclass
class PlayComparison:
    """A position where two configs disagree on the best play."""
    position_desc: str
    die1: int
    die2: int
    config_a: str
    config_b: str
    best_play_a: str
    best_play_b: str
    equity_a: float  # equity of config_a's best play under config_a's weights
    equity_b: float  # equity of config_b's best play under config_b's weights
    # What happens when we evaluate the OTHER config's choice:
    equity_a_of_b_play: float  # config_a's equity for config_b's best play
    equity_b_of_a_play: float  # config_b's equity for config_a's best play
    gap_a: float  # how much config_a loses by playing config_b's choice
    gap_b: float  # how much config_b loses by playing config_a's choice


CONFIGS = {
    "money":           {"weights": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0], "path": "output/money"},
    "dmp":             {"weights": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0], "path": "output/dmp"},
    "gammon_seeking":  {"weights": [1.0, 2.0, 2.0, -1.0, -1.0, -1.0], "path": "output/gammon_seeking"},
    "gammon_avoiding": {"weights": [1.0, 1.0, 1.0, -1.0, -2.0, -2.0], "path": "output/gammon_avoiding"},
}


def load_model(config_name: str) -> tuple[TDNetwork, torch.Tensor]:
    """Load the best checkpoint for a config and return (network, weights)."""
    cfg = CONFIGS[config_name]
    weights = torch.tensor(cfg["weights"], dtype=torch.float32)

    # Find the highest level checkpoint
    output_dir = Path(cfg["path"])
    checkpoints = sorted(output_dir.glob("checkpoint_level*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {output_dir}")

    # Pick the highest numbered non-final checkpoint
    best = None
    best_level = -1
    for cp in checkpoints:
        name = cp.stem
        if "final" in name:
            continue
        # Extract level number
        level = int(name.replace("checkpoint_level", ""))
        if level > best_level:
            best_level = level
            best = cp

    if best is None:
        best = checkpoints[-1]  # fallback to any checkpoint

    print(f"  {config_name}: loading {best.name} (level {best_level})")

    net = TDNetwork(input_size=BOARD_FEATURE_SIZE)
    data = torch.load(best, map_location="cpu", weights_only=True)
    net.load_state_dict(data["model_state_dict"])
    net.eval()

    return net, weights


def describe_play(play) -> str:
    """Human-readable description of a play."""
    if play.num_moves == 0:
        return "(no moves)"
    parts = []
    for m in play.moves:
        src = "bar" if m.source == 24 else str(m.source + 1)
        dst = "off" if m.dest == -1 else str(m.dest + 1)
        hit = "*" if m.hits else ""
        parts.append(f"{src}/{dst}{hit}")
    return " ".join(parts)


def describe_position(state: BoardState) -> str:
    """Board as int[26]: [0]=opponent bar, [1]-[24]=points, [25]=player bar."""
    board = [0] * 26
    board[25] = state.bar_player
    board[0] = -state.bar_opponent  # opponent's checkers are negative
    for i in range(24):
        board[i + 1] = int(state.points[i])
    return str(board)


def evaluate_plays(
    state: BoardState,
    plays: list,
    network: TDNetwork,
    weights: torch.Tensor,
) -> list[tuple[int, float]]:
    """Evaluate all plays and return sorted (index, equity) pairs."""
    if len(plays) == 0 or (len(plays) == 1 and plays[0].num_moves == 0):
        return [(0, 0.0)]

    features_list = []
    for play in plays:
        result_state = _apply_play(state, play)
        flipped = flip_perspective(result_state)
        features_list.append(encode_board(flipped))

    batch = torch.from_numpy(np.stack(features_list))
    with torch.no_grad():
        outputs = network(batch)
    equities = -compute_equity(outputs, weights)

    indexed = [(i, equities[i].item()) for i in range(len(plays))]
    indexed.sort(key=lambda x: -x[1])  # best first
    return indexed


def find_disagreements(
    positions: list[tuple[BoardState, int, int]],
    models: dict[str, tuple[TDNetwork, torch.Tensor]],
) -> dict[tuple[str, str], list[PlayComparison]]:
    """Find positions where model pairs disagree on the best play."""

    config_names = list(models.keys())
    # All 6 pairings
    pairs = []
    for i in range(len(config_names)):
        for j in range(i + 1, len(config_names)):
            pairs.append((config_names[i], config_names[j]))

    disagreements: dict[tuple[str, str], list[PlayComparison]] = {
        pair: [] for pair in pairs
    }

    for pos_idx, (state, d1, d2) in enumerate(positions):
        plays = generate_plays(state, d1, d2)
        if len(plays) <= 1:
            continue

        # Evaluate under each model
        evals = {}
        for name, (net, weights) in models.items():
            evals[name] = evaluate_plays(state, plays, net, weights)

        # Compare each pair
        for name_a, name_b in pairs:
            eval_a = evals[name_a]
            eval_b = evals[name_b]

            best_idx_a = eval_a[0][0]
            best_idx_b = eval_b[0][0]

            if best_idx_a == best_idx_b:
                continue  # same best play, no disagreement

            # Find equity of each model's best play under both weight sets
            equity_a_best = eval_a[0][1]
            equity_b_best = eval_b[0][1]

            # What does model_a think of model_b's best play?
            equity_a_of_b = next(eq for idx, eq in eval_a if idx == best_idx_b)
            # What does model_b think of model_a's best play?
            equity_b_of_a = next(eq for idx, eq in eval_b if idx == best_idx_a)

            gap_a = equity_a_best - equity_a_of_b  # how much A loses playing B's choice
            gap_b = equity_b_best - equity_b_of_a  # how much B loses playing A's choice

            total_gap = gap_a + gap_b

            comp = PlayComparison(
                position_desc=describe_position(state),
                die1=d1, die2=d2,
                config_a=name_a, config_b=name_b,
                best_play_a=describe_play(plays[best_idx_a]),
                best_play_b=describe_play(plays[best_idx_b]),
                equity_a=equity_a_best,
                equity_b=equity_b_best,
                equity_a_of_b_play=equity_a_of_b,
                equity_b_of_a_play=equity_b_of_a,
                gap_a=gap_a,
                gap_b=gap_b,
            )
            disagreements[(name_a, name_b)].append(comp)

    # Sort each pair's disagreements by total gap
    for pair in pairs:
        disagreements[pair].sort(key=lambda c: -(c.gap_a + c.gap_b))

    return disagreements


def main():
    parser = argparse.ArgumentParser(description="Compare equity weight configurations")
    parser.add_argument("--num-positions", type=int, default=500,
                        help="Number of positions to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading models...")
    models = {}
    for name in CONFIGS:
        try:
            net, weights = load_model(name)
            models[name] = (net, weights)
        except FileNotFoundError as e:
            print(f"  Skipping {name}: {e}")

    if len(models) < 2:
        print("Need at least 2 models to compare. Run training first.")
        return

    print(f"\nGenerating {args.num_positions} positions...")
    rng = np.random.default_rng(args.seed)
    gen = SetupGenerator(rng=rng)

    positions = []
    # Mix of Bg960, standard, and mid-game positions
    for i in range(args.num_positions):
        if i % 3 == 0:
            state = BoardState.standard_setup()
        elif i % 3 == 1:
            state = gen.generate()
        else:
            # Play a few random moves from standard to get mid-game positions
            state = BoardState.standard_setup()
            net, weights = list(models.values())[0]
            for _ in range(rng.integers(5, 20)):
                d1, d2 = roll_dice(rng)
                plays = generate_plays(state, d1, d2)
                if plays[0].num_moves == 0:
                    break
                idx = rng.integers(0, len(plays))
                state = _apply_play(state, plays[idx])
                state = flip_perspective(state)

        d1, d2 = roll_dice(rng)
        positions.append((state, int(d1), int(d2)))

    print(f"Finding disagreements across {len(positions)} positions...")
    t = time.perf_counter()
    disagreements = find_disagreements(positions, models)
    elapsed = time.perf_counter() - t
    print(f"Done in {elapsed:.1f}s\n")

    # Report top 3 for each pairing
    print("=" * 80)
    for (name_a, name_b), comps in disagreements.items():
        print(f"\n{name_a.upper()} vs {name_b.upper()}: {len(comps)} disagreements")
        print("-" * 80)

        if not comps:
            print("  No disagreements found.")
            continue

        for rank, comp in enumerate(comps[:3], 1):
            total_gap = comp.gap_a + comp.gap_b
            print(f"\n  #{rank} (total gap: {total_gap:.4f})")
            print(f"  Position: {comp.position_desc}")
            print(f"  Dice: {comp.die1}-{comp.die2}")
            print(f"  {comp.config_a:20s} plays: {comp.best_play_a}")
            print(f"    equity: {comp.equity_a:+.4f} (vs {comp.equity_a_of_b_play:+.4f} for other's play, gap: {comp.gap_a:.4f})")
            print(f"  {comp.config_b:20s} plays: {comp.best_play_b}")
            print(f"    equity: {comp.equity_b:+.4f} (vs {comp.equity_b_of_a_play:+.4f} for other's play, gap: {comp.gap_b:.4f})")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()