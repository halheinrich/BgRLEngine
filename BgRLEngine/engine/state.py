"""Board state representation and feature encoding.

Encodes a backgammon board position into a fixed-size feature vector
suitable for neural network input. The encoding is variant-agnostic:
all variants (standard, Nackgammon, bg960) use the same rules and
differ only in starting position.

Input vector layout (~303 features):
    Points (player):  24 points × 6 units = 144
    Points (opponent): 24 points × 6 units = 144
    Bar (player):      3 (thermometer: 1, 2, 3+)
    Bar (opponent):    3
    Borne off (player):  2 (count/15, all_off flag)
    Borne off (opponent): 2
    Player on roll:    1
    Pip count ratio:   1
    Race detected:     1
    Checker counts:    2 (player/15, opponent/15)
    ---
    Total board:       303

Cube/match features are appended separately when needed.
"""

from __future__ import annotations

import numpy as np

# Board constants
NUM_POINTS = 24
CHECKERS_PER_PLAYER = 15
UNITS_PER_POINT = 6
THERMOMETER_DEPTH = 5  # units 1-5 are binary thermometer
OVERFLOW_SCALE = 10.0  # unit 6: (n - THERMOMETER_DEPTH) / OVERFLOW_SCALE

# Feature vector sizes
POINT_FEATURES = NUM_POINTS * UNITS_PER_POINT  # 144 per player
BAR_FEATURES = 3  # per player
BORNE_OFF_FEATURES = 2  # per player
GLOBAL_FEATURES = 5  # on_roll, pip_ratio, race, checker_count_p, checker_count_o

BOARD_FEATURE_SIZE = (
    2 * POINT_FEATURES      # 288
    + 2 * BAR_FEATURES       # 6
    + 2 * BORNE_OFF_FEATURES  # 4
    + GLOBAL_FEATURES         # 5
)  # = 303

# Cube/match features (appended when relevant)
CUBE_FEATURES = 4  # log2(cube)/6, player_owns, opp_owns, centered
MATCH_FEATURES = 4  # player_away/N, opp_away/N, crawford, post_crawford
FULL_FEATURE_SIZE = BOARD_FEATURE_SIZE + CUBE_FEATURES + MATCH_FEATURES  # 311


class BoardState:
    """Represents a backgammon board position.

    Uses the standard convention:
        - points[0..23]: player's perspective, point 1-24
        - Positive values = player's checkers
        - Negative values = opponent's checkers
        - bar_player, bar_opponent: checkers on the bar
        - off_player, off_opponent: checkers borne off
        - player_to_move: True if it's the player's turn

    Point numbering: point index 0 = player's 1-point (bearing off destination),
    point index 23 = player's 24-point. Player moves from high to low indices.
    """

    __slots__ = (
        "points", "bar_player", "bar_opponent",
        "off_player", "off_opponent", "player_to_move",
    )

    def __init__(self) -> None:
        self.points: np.ndarray = np.zeros(NUM_POINTS, dtype=np.int16)
        self.bar_player: int = 0
        self.bar_opponent: int = 0
        self.off_player: int = 0
        self.off_opponent: int = 0
        self.player_to_move: bool = True

    def copy(self) -> BoardState:
        """Return a deep copy of this state."""
        new = BoardState()
        new.points = self.points.copy()
        new.bar_player = self.bar_player
        new.bar_opponent = self.bar_opponent
        new.off_player = self.off_player
        new.off_opponent = self.off_opponent
        new.player_to_move = self.player_to_move
        return new

    @staticmethod
    def from_setup(setup: np.ndarray) -> BoardState:
        """Create a BoardState from a starting setup array.

        Args:
            setup: array of length 24, positive = player checkers on that point,
                   negative = opponent checkers. Points indexed 0-23 where
                   index 0 = player's 1-point.

        Returns:
            A new BoardState with the given setup.
        """
        state = BoardState()
        state.points = np.array(setup, dtype=np.int16)
        state.bar_player = 0
        state.bar_opponent = 0
        state.off_player = 0
        state.off_opponent = 0
        state.player_to_move = True
        return state

    @staticmethod
    def standard_setup() -> BoardState:
        """Standard backgammon starting position."""
        setup = np.zeros(NUM_POINTS, dtype=np.int16)
        # Player's checkers (positive)
        setup[0] = 2    # 2 on the 1-point (opponent's home board = player's 24-pt)
        # Wait — let's use standard convention carefully.
        # Player moves from 24 → 1. Index 0 = point 1, index 23 = point 24.
        # Standard setup from player's perspective:
        #   point 6 (idx 5):  5 checkers
        #   point 8 (idx 7):  3 checkers
        #   point 13 (idx 12): 5 checkers
        #   point 24 (idx 23): 2 checkers
        setup = np.zeros(NUM_POINTS, dtype=np.int16)
        setup[5] = 5
        setup[7] = 3
        setup[12] = 5
        setup[23] = 2
        # Opponent's checkers (negative), mirrored:
        #   point 19 (idx 18): -5
        #   point 17 (idx 16): -3
        #   point 12 (idx 11): -5
        #   point 1 (idx 0):   -2
        setup[18] = -5
        setup[16] = -3
        setup[11] = -5
        setup[0] = -2
        return BoardState.from_setup(setup)

    @staticmethod
    def nackgammon_setup() -> BoardState:
        """Nackgammon starting position (Nack Ballard variant)."""
        setup = np.zeros(NUM_POINTS, dtype=np.int16)
        # Player's checkers:
        #   point 6 (idx 5):  4
        #   point 8 (idx 7):  3
        #   point 13 (idx 12): 4
        #   point 23 (idx 22): 2
        #   point 24 (idx 23): 2
        setup[5] = 4
        setup[7] = 3
        setup[12] = 4
        setup[22] = 2
        setup[23] = 2
        # Opponent mirrored:
        setup[18] = -4
        setup[16] = -3
        setup[11] = -4
        setup[1] = -2
        setup[0] = -2
        return BoardState.from_setup(setup)

    def player_pip_count(self) -> int:
        """Total pip count for the player (distance to bear off)."""
        pips = 0
        for i in range(NUM_POINTS):
            if self.points[i] > 0:
                pips += int(self.points[i]) * (i + 1)
        pips += self.bar_player * 25  # bar is 25 pips away
        return pips

    def opponent_pip_count(self) -> int:
        """Total pip count for the opponent."""
        pips = 0
        for i in range(NUM_POINTS):
            if self.points[i] < 0:
                pips += abs(int(self.points[i])) * (NUM_POINTS - i)
        pips += self.bar_opponent * 25
        return pips

    def is_race(self) -> bool:
        """True if no future contact is possible (pure race).

        Contact is impossible when no player checker is behind any opponent
        checker. Player moves high→low (24→1), opponent moves low→high (1→24).
        """
        if self.bar_player > 0 or self.bar_opponent > 0:
            return False

        # Find the lowest point (closest to bearing off) with opponent checkers
        # and the highest point with player checkers.
        lowest_opp = -1
        for i in range(NUM_POINTS):
            if self.points[i] < 0:
                lowest_opp = i
                break

        if lowest_opp == -1:
            return True  # no opponent checkers on board

        highest_player = -1
        for i in range(NUM_POINTS - 1, -1, -1):
            if self.points[i] > 0:
                highest_player = i
                break

        if highest_player == -1:
            return True  # no player checkers on board

        # Player moves toward index 0; opponent moves toward index 23.
        # Contact is impossible if all player checkers are at indices
        # lower than all opponent checkers.
        return highest_player < lowest_opp

    def player_checker_count(self) -> int:
        """Total player checkers in play (on board + bar)."""
        on_board = sum(max(0, self.points[i]) for i in range(NUM_POINTS))
        return on_board + self.bar_player

    def opponent_checker_count(self) -> int:
        """Total opponent checkers in play (on board + bar)."""
        on_board = sum(abs(min(0, self.points[i])) for i in range(NUM_POINTS))
        return on_board + self.bar_opponent


def encode_point(count: int) -> np.ndarray:
    """Encode a checker count using thermometer + overflow.

    Args:
        count: non-negative checker count on a point.

    Returns:
        Array of UNITS_PER_POINT (6) float values.
    """
    features = np.zeros(UNITS_PER_POINT, dtype=np.float32)
    for i in range(min(count, THERMOMETER_DEPTH)):
        features[i] = 1.0
    if count > THERMOMETER_DEPTH:
        features[THERMOMETER_DEPTH] = min(
            (count - THERMOMETER_DEPTH) / OVERFLOW_SCALE, 1.0
        )
    return features


def encode_bar(count: int) -> np.ndarray:
    """Encode bar checker count as thermometer (1, 2, 3+)."""
    features = np.zeros(BAR_FEATURES, dtype=np.float32)
    for i in range(min(count, BAR_FEATURES)):
        features[i] = 1.0
    return features


def encode_borne_off(count: int, total: int = CHECKERS_PER_PLAYER) -> np.ndarray:
    """Encode borne-off count as (fraction, all_off flag)."""
    return np.array(
        [count / total, 1.0 if count >= total else 0.0],
        dtype=np.float32,
    )


def encode_board(state: BoardState) -> np.ndarray:
    """Encode a BoardState into a feature vector.

    Returns:
        Float32 array of length BOARD_FEATURE_SIZE (303).
    """
    features = np.zeros(BOARD_FEATURE_SIZE, dtype=np.float32)
    idx = 0

    # Player points (24 × 6 = 144)
    for i in range(NUM_POINTS):
        count = max(0, state.points[i])
        features[idx:idx + UNITS_PER_POINT] = encode_point(count)
        idx += UNITS_PER_POINT

    # Opponent points (24 × 6 = 144)
    for i in range(NUM_POINTS):
        count = abs(min(0, state.points[i]))
        features[idx:idx + UNITS_PER_POINT] = encode_point(count)
        idx += UNITS_PER_POINT

    # Bar
    features[idx:idx + BAR_FEATURES] = encode_bar(state.bar_player)
    idx += BAR_FEATURES
    features[idx:idx + BAR_FEATURES] = encode_bar(state.bar_opponent)
    idx += BAR_FEATURES

    # Borne off
    features[idx:idx + BORNE_OFF_FEATURES] = encode_borne_off(state.off_player)
    idx += BORNE_OFF_FEATURES
    features[idx:idx + BORNE_OFF_FEATURES] = encode_borne_off(state.off_opponent)
    idx += BORNE_OFF_FEATURES

    # Global features
    features[idx] = 1.0 if state.player_to_move else 0.0
    idx += 1

    p_pips = state.player_pip_count()
    o_pips = state.opponent_pip_count()
    total_pips = p_pips + o_pips
    features[idx] = p_pips / total_pips if total_pips > 0 else 0.5
    idx += 1

    features[idx] = 1.0 if state.is_race() else 0.0
    idx += 1

    features[idx] = state.player_checker_count() / CHECKERS_PER_PLAYER
    idx += 1
    features[idx] = state.opponent_checker_count() / CHECKERS_PER_PLAYER
    idx += 1

    assert idx == BOARD_FEATURE_SIZE
    return features


def encode_cube(cube_value: int = 1, owner: int = 0) -> np.ndarray:
    """Encode cube state.

    Args:
        cube_value: current cube value (1, 2, 4, 8, ...)
        owner: 0 = centered, 1 = player owns, -1 = opponent owns

    Returns:
        Float32 array of length CUBE_FEATURES (4).
    """
    import math
    features = np.zeros(CUBE_FEATURES, dtype=np.float32)
    features[0] = math.log2(cube_value) / 6.0  # normalize: 64 → 1.0
    features[1] = 1.0 if owner == 1 else 0.0
    features[2] = 1.0 if owner == -1 else 0.0
    features[3] = 1.0 if owner == 0 else 0.0
    return features


def encode_match(
    player_away: int,
    opponent_away: int,
    match_length: int,
    crawford: bool = False,
    post_crawford: bool = False,
) -> np.ndarray:
    """Encode match score state.

    Args:
        player_away: points player needs to win the match.
        opponent_away: points opponent needs.
        match_length: total match length.
        crawford: True if this is the Crawford game.
        post_crawford: True if Crawford game has passed.

    Returns:
        Float32 array of length MATCH_FEATURES (4).
    """
    features = np.zeros(MATCH_FEATURES, dtype=np.float32)
    features[0] = player_away / match_length if match_length > 0 else 0.0
    features[1] = opponent_away / match_length if match_length > 0 else 0.0
    features[2] = 1.0 if crawford else 0.0
    features[3] = 1.0 if post_crawford else 0.0
    return features


def encode_full(
    state: BoardState,
    cube_value: int = 1,
    cube_owner: int = 0,
    player_away: int = 0,
    opponent_away: int = 0,
    match_length: int = 0,
    crawford: bool = False,
    post_crawford: bool = False,
) -> np.ndarray:
    """Encode a complete game state including board, cube, and match.

    Returns:
        Float32 array of length FULL_FEATURE_SIZE (311).
    """
    board = encode_board(state)
    cube = encode_cube(cube_value, cube_owner)
    match = encode_match(player_away, opponent_away, match_length,
                         crawford, post_crawford)
    return np.concatenate([board, cube, match])


def flip_perspective(state: BoardState) -> BoardState:
    """Return a new BoardState with player/opponent swapped.

    This is used so that the network always evaluates from the
    perspective of the player on roll.
    """
    flipped = BoardState()
    # Reverse and negate the points array
    flipped.points = -state.points[::-1].copy()
    flipped.bar_player = state.bar_opponent
    flipped.bar_opponent = state.bar_player
    flipped.off_player = state.off_opponent
    flipped.off_opponent = state.off_player
    flipped.player_to_move = not state.player_to_move
    return flipped
