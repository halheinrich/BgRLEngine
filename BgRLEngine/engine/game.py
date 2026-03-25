"""Game simulation for self-play training.

Plays complete backgammon games using a neural network for move selection.
Collects state transitions for TD(λ) training.

Move generation is delegated to BgMoveGen via generate_successor_states().
Each successor is already from the perspective of the player now on roll.
A pass (no legal moves) is returned as a single successor — the flipped
current state — and is handled identically to a forced move.
"""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from engine.state import BoardState, encode_board
from engine.dice import roll_dice
from engine.movegen import generate_successor_states
from engine.network import TDNetwork, compute_equity


class GameResult(IntEnum):
    """Outcome of a game from the initial player's perspective."""
    LOSE_BACKGAMMON = -3
    LOSE_GAMMON     = -2
    LOSE            = -1
    IN_PROGRESS     =  0
    WIN             =  1
    WIN_GAMMON      =  2
    WIN_BACKGAMMON  =  3


@dataclass
class GameRecord:
    """Record of a completed game for training.

    Stores the sequence of board states (as feature vectors) and the
    final outcome for TD(λ) updates.

    All states are encoded from the initial player's perspective.
    The result is also from the initial player's perspective.
    """
    states:    list[np.ndarray] = field(default_factory=list)
    result:    GameResult       = GameResult.IN_PROGRESS
    num_moves: int              = 0


def _determine_result(state: BoardState, player_is_initial: bool) -> GameResult:
    """Determine game result if a player has borne off all checkers.

    Args:
        state: board state to check (on-roll player's perspective).
        player_is_initial: True if the on-roll player is the initial player.

    Returns:
        GameResult, or IN_PROGRESS if the game isn't over.
    """
    if state.off_player >= 15:
        if state.off_opponent == 0:
            has_checker_in_home = (
                state.bar_opponent > 0
                or any(state.points[i] < 0 for i in range(6))
            )
            result = (
                GameResult.WIN_BACKGAMMON if has_checker_in_home
                else GameResult.WIN_GAMMON
            )
        else:
            result = GameResult.WIN

        if not player_is_initial:
            result = GameResult(-result)

        return result

    return GameResult.IN_PROGRESS


def select_play(
    successors:     list[BoardState],
    network:        TDNetwork,
    device:         torch.device,
    epsilon:        float                    = 0.0,
    rng:            Optional[np.random.Generator] = None,
    equity_weights: Optional[torch.Tensor]   = None,
) -> BoardState:
    """Select the best successor state using the neural network.

    Successor states arrive already from the perspective of the player
    now on roll — no flipping needed before encoding.

    A single successor (including a forced move or pass) is returned
    immediately without network evaluation.

    Uses ε-greedy: with probability ε select randomly, otherwise select
    the successor with the highest equity in a single batched forward pass.

    Args:
        successors:     list of successor BoardStates from BgMoveGen.
        network:        the evaluation network.
        device:         torch device.
        epsilon:        exploration rate.
        rng:            random number generator.
        equity_weights: optional equity weights tensor.

    Returns:
        Selected BoardState.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Single successor: forced move or pass — no evaluation needed
    if len(successors) == 1:
        return successors[0]

    # ε-greedy exploration
    if epsilon > 0 and rng.random() < epsilon:
        return successors[rng.integers(0, len(successors))]

    # Encode all successors — already on-roll player's perspective
    features_list = [encode_board(s) for s in successors]

    # Batched forward pass
    batch = torch.from_numpy(np.stack(features_list)).to(device)
    with torch.no_grad():
        outputs = network(batch)

    # Negate: successors are from next player's perspective; we want
    # the move that minimises their equity (maximises ours)
    equities = -compute_equity(outputs, equity_weights)
    best_idx = equities.argmax().item()

    return successors[best_idx]


def play_game(
    network:        TDNetwork,
    device:         torch.device,
    starting_state: Optional[BoardState]          = None,
    opponent:       Optional[TDNetwork]            = None,
    epsilon:        float                          = 0.0,
    rng:            Optional[np.random.Generator]  = None,
    max_moves:      int                            = 1000,
    equity_weights: Optional[torch.Tensor]         = None,
) -> GameRecord:
    """Play a complete game via self-play or against an opponent.

    If opponent is None, both sides use the same network (self-play).
    If opponent is provided, the initial player uses `network` and
    the second player uses `opponent`.

    All states are recorded from the initial player's perspective.
    BgMoveGen ensures every state is always from the on-roll player's
    perspective — no perspective flipping in this file.

    Args:
        network:        the evaluation network (plays as initial player).
        device:         torch device.
        starting_state: initial board state (default: standard setup).
        opponent:       optional separate network for the second player.
        epsilon:        exploration rate for ε-greedy.
        rng:            random number generator.
        max_moves:      safety limit on game length.
        equity_weights: optional equity weights for move selection.

    Returns:
        GameRecord with state sequence and final result.
    """
    if rng is None:
        rng = np.random.default_rng()
    if starting_state is None:
        starting_state = BoardState.standard_setup()

    state  = starting_state.copy()
    record = GameRecord()
    initial_player_turn = True

    for _ in range(max_moves):
        # Record current state from initial player's perspective.
        # When it's the initial player's turn, state is already in their view.
        # When it's the opponent's turn, state is in opponent's view — encode
        # the negated/reversed board for the record.
        record.states.append(encode_board(state) if initial_player_turn
                              else encode_board(_encode_as_initial(state)))

        # Roll dice
        die1, die2 = roll_dice(rng)

        # Get successor states from BgMoveGen.
        # Always at least one (pass = flipped current state).
        successors = generate_successor_states(state, die1, die2)

        # Select network for this turn
        active_net = network if (opponent is None or initial_player_turn) else opponent

        # Select best successor
        new_state = select_play(
            successors, active_net, device, epsilon, rng, equity_weights
        )

        record.num_moves += 1

        # Check for game over.
        # new_state is from the perspective of the player now on roll
        # (i.e. the opponent of whoever just moved). A win for the player
        # who just moved shows as off_opponent >= 15 in new_state.
        result = _determine_result_from_successor(new_state, initial_player_turn)
        if result != GameResult.IN_PROGRESS:
            record.result = result
            record.states.append(encode_board(new_state) if not initial_player_turn
                                  else encode_board(_encode_as_initial(new_state)))
            return record

        state = new_state
        initial_player_turn = not initial_player_turn

    # Safety limit reached
    record.result = GameResult.WIN
    return record


def _encode_as_initial(state: BoardState) -> BoardState:
    """Return a view of state from the initial player's perspective.

    Called only when it is the opponent's turn and we need to record
    the state from the initial player's viewpoint for the training record.
    This is the only place perspective adjustment occurs in this file.
    """
    flipped = BoardState()
    flipped.points       = -state.points[::-1].copy()
    flipped.bar_player   = state.bar_opponent
    flipped.bar_opponent = state.bar_player
    flipped.off_player   = state.off_opponent
    flipped.off_opponent = state.off_player
    return flipped


def _determine_result_from_successor(
    new_state:          BoardState,
    initial_player_turn: bool,
) -> GameResult:
    """Check for game over in the successor state.

    new_state is from the perspective of the player now on roll —
    the opponent of whoever just moved. A win for the player who just
    moved appears as off_opponent >= 15 in new_state.
    """
    if new_state.off_opponent >= 15:
        # The player who just moved has won
        if new_state.off_player == 0:
            has_checker_in_home = (
                new_state.bar_player > 0
                or any(new_state.points[i] > 0 for i in range(6))
            )
            result = (
                GameResult.WIN_BACKGAMMON if has_checker_in_home
                else GameResult.WIN_GAMMON
            )
        else:
            result = GameResult.WIN

        # The player who just moved: initial player if initial_player_turn
        if not initial_player_turn:
            result = GameResult(-result)

        return result

    return GameResult.IN_PROGRESS