"""Dice rolling and legal move generation.

Handles standard backgammon dice rules:
    - Two dice, values 1-6
    - Doubles: use each die twice (4 moves instead of 2)
    - Must use both dice if possible; if only one can be used, must use the larger
    - When bearing off, can use a higher die to bear off a checker on a lower point
      if no checker exists on the exact point
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from dataclasses import dataclass

from engine.state import BoardState, NUM_POINTS


@dataclass(frozen=True)
class Move:
    """A single checker move from one point to another.

    Attributes:
        source: source point index (0-23), or 24 for bar entry.
        dest: destination point index (0-23), or -1 for bear off.
        die: die value used for this move.
        hits: True if this move hits an opponent blot.
    """
    source: int
    dest: int
    die: int
    hits: bool = False


@dataclass
class Play:
    """A complete play (sequence of moves) for one turn.

    A play consists of 1-4 individual moves using the rolled dice.
    """
    moves: list[Move]

    @property
    def num_moves(self) -> int:
        return len(self.moves)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Play):
            return NotImplemented
        return self.moves == other.moves

    def __hash__(self) -> int:
        return hash(tuple(self.moves))


def roll_dice(rng: Optional[np.random.Generator] = None) -> tuple[int, int]:
    """Roll two dice.

    Returns:
        Tuple of (die1, die2), each 1-6.
    """
    if rng is None:
        rng = np.random.default_rng()
    return int(rng.integers(1, 7)), int(rng.integers(1, 7))


def get_dice_to_use(die1: int, die2: int) -> list[int]:
    """Get the list of dice values to use.

    Doubles give four uses of the same value.

    Returns:
        List of die values to use (2 or 4 elements).
    """
    if die1 == die2:
        return [die1] * 4
    return [die1, die2]


def _can_bear_off(state: BoardState) -> bool:
    """Check if the player can bear off (all checkers in home board).

    Home board = points 1-6, indices 0-5.
    """
    if state.bar_player > 0:
        return False
    for i in range(6, NUM_POINTS):
        if state.points[i] > 0:
            return False
    return True


def _single_moves(state: BoardState, die: int) -> list[Move]:
    """Generate all legal single-checker moves for a given die value.

    Args:
        state: current board state (from player's perspective).
        die: die value to use.

    Returns:
        List of legal Move objects.
    """
    moves = []

    # Must enter from bar first
    if state.bar_player > 0:
        dest = NUM_POINTS - die  # entering from the opponent's home board
        if dest < 0 or dest >= NUM_POINTS:
            return []  # invalid (shouldn't happen with die 1-6)
        point_val = state.points[dest]
        if point_val >= -1:  # empty, own checkers, or single opponent (blot)
            hits = point_val == -1
            moves.append(Move(source=24, dest=dest, die=die, hits=hits))
        return moves

    # Regular moves and bearing off
    bear_off = _can_bear_off(state)

    for src in range(NUM_POINTS):
        if state.points[src] <= 0:
            continue  # no player checkers here

        dest = src - die  # player moves toward index 0

        if dest >= 0:
            # Regular move
            point_val = state.points[dest]
            if point_val >= -1:
                hits = point_val == -1
                moves.append(Move(source=src, dest=dest, die=die, hits=hits))
        elif bear_off:
            # Bearing off
            if dest == -1:
                # Exact roll to bear off
                moves.append(Move(source=src, dest=-1, die=die))
            elif dest < -1:
                # Overshoot: only allowed if no checker on a higher point
                higher_exists = any(
                    state.points[j] > 0 for j in range(src + 1, 6)
                )
                if not higher_exists:
                    moves.append(Move(source=src, dest=-1, die=die))

    return moves


def _apply_move(state: BoardState, move: Move) -> BoardState:
    """Apply a single move to a board state.

    Returns a new BoardState (does not modify the input).
    """
    new = state.copy()

    # Remove checker from source
    if move.source == 24:
        new.bar_player -= 1
    else:
        new.points[move.source] -= 1

    # Place checker at destination
    if move.dest == -1:
        new.off_player += 1
    else:
        if move.hits:
            new.points[move.dest] = 0  # remove opponent blot
            new.bar_opponent += 1
        new.points[move.dest] += 1

    return new


def generate_plays(state: BoardState, die1: int, die2: int) -> list[Play]:
    """Generate all legal complete plays for a dice roll.

    Enforces the rule that both dice must be used if possible, and
    if only one can be used, the larger die must be used.

    Args:
        state: current board state.
        die1: first die value.
        die2: second die value.

    Returns:
        List of unique legal Play objects.
    """
    dice = get_dice_to_use(die1, die2)
    is_doubles = die1 == die2

    if is_doubles:
        plays = _generate_doubles_plays(state, dice)
    else:
        plays = _generate_regular_plays(state, dice)

    # Deduplicate plays (different orderings can produce same result)
    seen: set[tuple[tuple[int, int], ...]] = set()
    unique_plays: list[Play] = []
    for play in plays:
        # Normalize: represent by sorted (source, dest) pairs
        key = tuple(sorted((m.source, m.dest) for m in play.moves))
        if key not in seen:
            seen.add(key)
            unique_plays.append(play)

    return unique_plays if unique_plays else [Play(moves=[])]  # empty play = no legal moves


def _generate_regular_plays(
    state: BoardState, dice: list[int]
) -> list[Play]:
    """Generate plays for non-doubles (two distinct dice).

    Try both orderings and enforce the "use both if possible,
    otherwise use the larger die" rule.
    """
    all_plays: list[Play] = []

    # Try both orderings of the dice
    for ordered_dice in [(dice[0], dice[1]), (dice[1], dice[0])]:
        _recursive_generate(state, list(ordered_dice), [], all_plays)

    if not all_plays:
        return []

    # Find the maximum number of dice used
    max_dice_used = max(p.num_moves for p in all_plays)

    if max_dice_used == 0:
        return []

    # Filter to plays using the maximum number of dice
    best_plays = [p for p in all_plays if p.num_moves == max_dice_used]

    # If only one die can be used, must use the larger one
    if max_dice_used == 1:
        max_die = max(dice)
        plays_with_max = [p for p in best_plays if p.moves[0].die == max_die]
        if plays_with_max:
            return plays_with_max

    return best_plays


def _generate_doubles_plays(
    state: BoardState, dice: list[int]
) -> list[Play]:
    """Generate plays for doubles (use up to 4 of the same die)."""
    all_plays: list[Play] = []
    _recursive_generate(state, dice, [], all_plays)

    if not all_plays:
        return []

    # Must use as many dice as possible
    max_dice_used = max(p.num_moves for p in all_plays)
    return [p for p in all_plays if p.num_moves == max_dice_used]


def _recursive_generate(
    state: BoardState,
    remaining_dice: list[int],
    current_moves: list[Move],
    all_plays: list[Play],
) -> None:
    """Recursively generate all possible move sequences.

    Args:
        state: current board state after moves so far.
        remaining_dice: dice values still to use.
        current_moves: moves made so far in this play.
        all_plays: accumulator for complete plays.
    """
    if not remaining_dice:
        all_plays.append(Play(moves=list(current_moves)))
        return

    die = remaining_dice[0]
    legal = _single_moves(state, die)

    if not legal:
        # Can't use this die; record what we have so far
        all_plays.append(Play(moves=list(current_moves)))
        return

    for move in legal:
        new_state = _apply_move(state, move)
        current_moves.append(move)
        _recursive_generate(new_state, remaining_dice[1:], current_moves, all_plays)
        current_moves.pop()
