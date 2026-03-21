"""Game simulation for self-play training.

Plays complete backgammon games using a neural network for move selection.
Collects state transitions for TD(λ) training.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from engine.state import BoardState, encode_board, flip_perspective
from engine.dice import roll_dice, generate_plays, Play, _apply_move
from engine.network import TDNetwork, compute_equity


class GameResult(IntEnum):
    """Outcome of a game from the initial player's perspective."""
    LOSE_BACKGAMMON = -3
    LOSE_GAMMON = -2
    LOSE = -1
    IN_PROGRESS = 0
    WIN = 1
    WIN_GAMMON = 2
    WIN_BACKGAMMON = 3


@dataclass
class GameRecord:
    """Record of a completed game for training.

    Stores the sequence of board states (as feature vectors) and the
    final outcome for TD(λ) updates.

    All states are encoded from the initial player's perspective.
    The result is also from the initial player's perspective.
    """
    states: list[np.ndarray] = field(default_factory=list)
    result: GameResult = GameResult.IN_PROGRESS
    num_moves: int = 0


def _determine_result(state: BoardState, player_is_initial: bool) -> GameResult:
    """Determine game result if a player has borne off all checkers.

    Args:
        state: board state to check.
        player_is_initial: True if the current 'player' in the state
                          is the initial player (the one we're tracking).

    Returns:
        GameResult, or IN_PROGRESS if the game isn't over.
    """
    # Check if current player has won
    if state.off_player >= 15:
        # Current player won. Determine gammon/backgammon.
        if state.off_opponent == 0:
            # Opponent hasn't borne off any
            # Check for backgammon: opponent has checker on bar or in winner's home board
            has_checker_in_home = (
                state.bar_opponent > 0
                or any(state.points[i] < 0 for i in range(6))
            )
            result = GameResult.WIN_BACKGAMMON if has_checker_in_home else GameResult.WIN_GAMMON
        else:
            result = GameResult.WIN

        if not player_is_initial:
            # The winner is the opponent of the initial player
            result = GameResult(-result)

        return result

    return GameResult.IN_PROGRESS


def select_play(
    state: BoardState,
    plays: list[Play],
    network: TDNetwork,
    device: torch.device,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    equity_weights: Optional[torch.Tensor] = None,
) -> tuple[Play, BoardState]:
    """Select the best play using the neural network.

    Uses ε-greedy: with probability ε, select a random play;
    otherwise, select the play with the highest equity.
    Evaluates all candidate positions in a single batched forward pass.

    Args:
        state: current board state.
        plays: list of legal plays.
        network: the evaluation network.
        device: torch device.
        epsilon: exploration rate.
        rng: random number generator.
        equity_weights: optional equity weights tensor (default: money play).

    Returns:
        Tuple of (selected Play, resulting BoardState).
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(plays) == 1:
        # Only one play (including forced moves and no-move)
        result_state = _apply_play(state, plays[0])
        return plays[0], result_state

    # ε-greedy exploration
    if epsilon > 0 and rng.random() < epsilon:
        idx = rng.integers(0, len(plays))
        result_state = _apply_play(state, plays[idx])
        return plays[idx], result_state

    # Apply all plays and collect resulting states
    result_states = [_apply_play(state, play) for play in plays]

    # Encode all resulting positions (flipped, since opponent moves next)
    features_list = []
    for rs in result_states:
        flipped = flip_perspective(rs)
        features_list.append(encode_board(flipped))

    # Batch evaluate in a single forward pass
    batch = torch.from_numpy(np.stack(features_list)).to(device)
    with torch.no_grad():
        outputs = network(batch)
    # Equity from opponent's perspective; negate for current player
    equities = -compute_equity(outputs, equity_weights)

    # Pick the best
    best_idx = equities.argmax().item()
    return plays[best_idx], result_states[best_idx]


def _evaluate_position(
    state: BoardState,
    network: TDNetwork,
    device: torch.device,
) -> float:
    """Evaluate a position from the current player's perspective.

    Flips perspective so the network always evaluates from the
    mover's point of view, then computes equity.
    """
    # After a play, it's the opponent's turn. We want to evaluate
    # from the perspective of the player who just moved.
    # The network evaluates from the perspective of the player on roll.
    # So we flip to get the opponent's view, then negate the equity.
    flipped = flip_perspective(state)
    features = encode_board(flipped)
    tensor = torch.from_numpy(features).to(device)
    output = network.evaluate(tensor)
    # Equity from opponent's perspective; negate for current player
    return -compute_equity(output).item()


def _apply_play(state: BoardState, play: Play) -> BoardState:
    """Apply a complete play (sequence of moves) to a board state."""
    current = state
    for move in play.moves:
        current = _apply_move(current, move)
    return current


def play_game(
    network: TDNetwork,
    device: torch.device,
    starting_state: Optional[BoardState] = None,
    opponent: Optional[TDNetwork] = None,
    epsilon: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    max_moves: int = 1000,
    equity_weights: Optional[torch.Tensor] = None,
) -> GameRecord:
    """Play a complete game via self-play or against an opponent.

    If opponent is None, both sides use the same network (self-play).
    If opponent is provided, the initial player uses `network` and
    the second player uses `opponent`.

    All states are recorded from the initial player's perspective.

    Args:
        network: the evaluation network (plays as initial player).
        device: torch device.
        starting_state: initial board state (default: standard setup).
        opponent: optional separate network for the second player.
        epsilon: exploration rate for ε-greedy.
        rng: random number generator.
        max_moves: safety limit on game length.
        equity_weights: optional equity weights for move selection.

    Returns:
        GameRecord with state sequence and final result.
    """
    if rng is None:
        rng = np.random.default_rng()

    if starting_state is None:
        starting_state = BoardState.standard_setup()

    state = starting_state.copy()
    record = GameRecord()
    initial_player_turn = True  # track whose turn it is relative to initial player

    for move_num in range(max_moves):
        # Always encode from the initial player's perspective for training.
        # When it's the initial player's turn, state is already in their view.
        # When it's the opponent's turn, state is in opponent's view — flip it.
        if initial_player_turn:
            features = encode_board(state)
        else:
            features = encode_board(flip_perspective(state))
        record.states.append(features)

        # Roll dice
        die1, die2 = roll_dice(rng)

        # Generate legal plays
        plays = generate_plays(state, die1, die2)

        # Pick which network to use for move selection
        if opponent is None:
            active_net = network  # self-play
        else:
            active_net = network if initial_player_turn else opponent

        # Select and apply a play
        play, new_state = select_play(
            state, plays, active_net, device, epsilon, rng, equity_weights
        )
        record.num_moves += 1

        # Check for game over
        result = _determine_result(new_state, initial_player_turn)
        if result != GameResult.IN_PROGRESS:
            record.result = result
            # Record the final state from initial player's perspective
            if initial_player_turn:
                record.states.append(encode_board(new_state))
            else:
                record.states.append(encode_board(flip_perspective(new_state)))
            return record

        # Switch perspective for next turn
        state = flip_perspective(new_state)
        initial_player_turn = not initial_player_turn

    # Game exceeded max moves — treat as a draw / plain win for whoever is ahead
    record.result = GameResult.WIN  # fallback
    return record
