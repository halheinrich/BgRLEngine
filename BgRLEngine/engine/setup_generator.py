"""Bg960 starting position generator.

Generates random starting positions for backgammon with configurable
constraints. All setups use standard backgammon rules; only the
starting checker placement varies.

Constraints:
    - Symmetrical: opponent mirrors player setup
    - No checkers on bar or borne off
    - At least 2 checkers on every occupied point (no blots)
    - At least one occupied point per quadrant
    - Minimum pip count threshold
    - Number of made points sampled from configurable weighted distribution
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from engine.state import BoardState, NUM_POINTS, CHECKERS_PER_PLAYER

# Quadrant boundaries (0-indexed point indices)
QUADRANTS = [
    range(0, 6),    # home board (points 1-6)
    range(6, 12),   # outer board (points 7-12)
    range(12, 18),  # opponent outer board (points 13-18)
    range(18, 24),  # opponent home board (points 19-24)
]

# Default made-point weights
DEFAULT_MADE_POINT_WEIGHTS = {
    2: 1,
    3: 3,
    4: 10,
    5: 10,
    6: 5,
    7: 2,
}


class SetupGenerator:
    """Generates random backgammon starting positions.

    Positions satisfy the bg960 constraints and can be used directly
    with BoardState.from_setup(). The generator produces only the
    player's setup; the opponent's setup is the mirror image.

    Args:
        min_checkers_per_point: minimum checkers on any occupied point.
        min_points_per_quadrant: minimum occupied points per quadrant.
        min_pip_count: minimum total pip count for the player.
        checkers_per_player: total checkers to place.
        made_point_weights: dict mapping num_made_points → sampling weight.
        rng: numpy random Generator (or None for default).
    """

    def __init__(
        self,
        min_checkers_per_point: int = 2,
        min_points_per_quadrant: int = 1,
        min_pip_count: int = 100,
        checkers_per_player: int = CHECKERS_PER_PLAYER,
        made_point_weights: Optional[dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.min_checkers = min_checkers_per_point
        self.min_per_quadrant = min_points_per_quadrant
        self.min_pips = min_pip_count
        self.total_checkers = checkers_per_player
        self.weights = made_point_weights or DEFAULT_MADE_POINT_WEIGHTS
        self.rng = rng or np.random.default_rng()

        # Precompute valid made-point counts and their probabilities
        self._valid_counts: list[int] = []
        self._probs: np.ndarray = np.array([], dtype=np.float64)
        self._update_distribution()

    def _update_distribution(self) -> None:
        """Recompute sampling distribution from current weights."""
        # Filter to counts that are feasible given checker constraints
        max_points = self.total_checkers // self.min_checkers
        counts = []
        weights = []
        for k, w in sorted(self.weights.items()):
            if k >= 4 and k <= max_points:  # need at least 4 for quadrant constraint
                counts.append(k)
                weights.append(w)
        if not counts:
            raise ValueError(
                f"No valid made-point counts: need at least 4 points "
                f"(one per quadrant) and at most {max_points} "
                f"({self.total_checkers} checkers / {self.min_checkers} min each)"
            )
        self._valid_counts = counts
        w = np.array(weights, dtype=np.float64)
        self._probs = w / w.sum()

    def set_weights(self, weights: dict[int, float]) -> None:
        """Update made-point weights (can be changed between training runs)."""
        self.weights = weights
        self._update_distribution()

    def _sample_num_points(self) -> int:
        """Sample how many points to occupy."""
        return self.rng.choice(self._valid_counts, p=self._probs)

    def _select_points(self, num_points: int) -> list[int]:
        """Select which points to occupy, ensuring quadrant coverage.

        Also ensures no two selected points are mirrors of each other
        (i.e., point i and point 23-i are never both selected), so
        player and opponent checkers don't collide.

        Returns:
            Sorted list of point indices (0-23).
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            # Ensure at least one point per quadrant
            mandatory = []
            blocked: set[int] = set()
            
            for q in QUADRANTS:
                # Choose from points not blocked by mirrors of already-chosen points
                candidates = [p for p in q if p not in blocked]
                if not candidates:
                    break  # retry
                pt = int(self.rng.choice(candidates))
                mandatory.append(pt)
                blocked.add(pt)
                blocked.add(NUM_POINTS - 1 - pt)  # block the mirror
            else:
                # All quadrants satisfied
                mandatory = list(set(mandatory))

                if num_points < len(mandatory):
                    continue

                # Fill remaining points from non-blocked, non-selected points
                remaining_slots = num_points - len(mandatory)
                available = [
                    p for p in range(NUM_POINTS)
                    if p not in blocked
                ]
                if remaining_slots > len(available):
                    continue

                if remaining_slots > 0:
                    extra_pts = []
                    avail_set = set(available)
                    for _ in range(remaining_slots):
                        if not avail_set:
                            break
                        pt = int(self.rng.choice(list(avail_set)))
                        extra_pts.append(pt)
                        avail_set.discard(pt)
                        avail_set.discard(NUM_POINTS - 1 - pt)
                    if len(extra_pts) < remaining_slots:
                        continue
                    mandatory.extend(extra_pts)

                points = sorted(mandatory)
                return points

        raise RuntimeError(
            f"Failed to select {num_points} points satisfying quadrant constraint"
        )

    def _distribute_checkers(self, points: list[int]) -> np.ndarray:
        """Distribute checkers across the selected points.

        Each point gets at least min_checkers. The remainder is distributed
        uniformly at random.

        Returns:
            Array of length 24 with checker counts on selected points.
        """
        k = len(points)
        base = self.min_checkers * k
        remainder = self.total_checkers - base

        if remainder < 0:
            raise ValueError(
                f"Cannot place {self.min_checkers} checkers on each of "
                f"{k} points with only {self.total_checkers} total checkers"
            )

        # Distribute remainder using a random partition
        # (uniform over compositions via the "balls in bins" method)
        if remainder == 0:
            extra = np.zeros(k, dtype=np.int32)
        else:
            # Random composition: place remainder into k bins
            # Using the "stars and bars" sampling with sorting
            dividers = np.sort(
                self.rng.integers(0, remainder + 1, size=k - 1)
            )
            dividers = np.concatenate([[0], dividers, [remainder]])
            extra = np.diff(dividers).astype(np.int32)

        setup = np.zeros(NUM_POINTS, dtype=np.int16)
        for i, pt in enumerate(points):
            setup[pt] = self.min_checkers + extra[i]

        return setup

    def _compute_pip_count(self, setup: np.ndarray) -> int:
        """Compute player pip count from setup array."""
        pips = 0
        for i in range(NUM_POINTS):
            if setup[i] > 0:
                pips += int(setup[i]) * (i + 1)
        return pips

    def _mirror_setup(self, player_setup: np.ndarray) -> np.ndarray:
        """Create the full board with opponent as mirror of player.

        The opponent's checkers are placed symmetrically: if the player
        has checkers on point i (0-indexed), the opponent has checkers
        on point (23 - i), stored as negative values.

        Returns:
            Array of length 24 with both player (positive) and opponent
            (negative) checker counts.
        """
        # Build opponent array first, then combine
        opponent = np.zeros(NUM_POINTS, dtype=np.int16)
        for i in range(NUM_POINTS):
            if player_setup[i] > 0:
                mirror_idx = NUM_POINTS - 1 - i
                opponent[mirror_idx] += player_setup[i]

        # Combine: positive = player, negative = opponent
        full = player_setup.astype(np.int16).copy()
        for i in range(NUM_POINTS):
            if opponent[i] > 0:
                full[i] -= opponent[i]
        return full

    def generate(self, max_attempts: int = 10000) -> BoardState:
        """Generate a single random starting position.

        Retries until all constraints are satisfied (particularly pip count).

        Args:
            max_attempts: maximum generation attempts before raising.

        Returns:
            A BoardState with the random starting position.

        Raises:
            RuntimeError: if no valid position found within max_attempts.
        """
        for _ in range(max_attempts):
            num_points = self._sample_num_points()
            points = self._select_points(num_points)
            player_setup = self._distribute_checkers(points)

            # Check pip count
            pips = self._compute_pip_count(player_setup)
            if pips < self.min_pips:
                continue

            # Create full board with opponent mirror
            full_setup = self._mirror_setup(player_setup)

            return BoardState.from_setup(full_setup)

        raise RuntimeError(
            f"Failed to generate valid setup in {max_attempts} attempts"
        )

    def generate_batch(self, n: int) -> list[BoardState]:
        """Generate n random starting positions."""
        return [self.generate() for _ in range(n)]

    def standard(self) -> BoardState:
        """Return the standard backgammon starting position."""
        return BoardState.standard_setup()

    def nackgammon(self) -> BoardState:
        """Return the Nackgammon starting position."""
        return BoardState.nackgammon_setup()
