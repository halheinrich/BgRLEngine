"""TD(λ) training loop for self-play.

Implements the core training algorithm:
1. Play a game via self-play on CPU (fast for small networks)
2. Move network to GPU, compute TD(λ) updates via backprop
3. Move network back to CPU for next game

The training loop also handles:
- ε-greedy exploration with linear decay
- Periodic evaluation against frozen checkpoints
- SPRT-based level promotion
- Plateau detection and training termination

Device strategy:
- Self-play and evaluation run on CPU (avoids CUDA kernel launch overhead
  which dominates for small networks with per-move forward passes)
- TD(λ) backpropagation runs on GPU when available (benefits from
  parallelism in the backward pass across the game's state sequence)
- Frozen opponent checkpoints always stay on CPU
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from engine.state import BOARD_FEATURE_SIZE
from engine.network import TDNetwork, NUM_OUTPUTS, compute_equity
from engine.game import play_game, GameRecord, GameResult
from engine.movegen import get_starting_position, Variant
from utils.sprt import SPRTResult, sprt_test

@dataclass
class TrainingStats:
    """Accumulated training statistics."""
    games_played:          int   = 0
    total_moves:           int   = 0
    current_level:         int   = 0
    levels_reached:        int   = 0
    best_win_rate:         float = 0.0
    sprt_tests_run:        int   = 0
    sprt_tests_passed:     int   = 0
    sprt_tests_failed:     int   = 0
    games_since_level_up:  int   = 0
    training_start_time:   float = 0.0
    rolling_win_rate:      float = 0.5
    rolling_eval_history:  list[float] = field(default_factory=list)


def result_to_target(result: GameResult) -> np.ndarray:
    """Convert a game result to the target output vector.

    Maps the final outcome to the 6 output probabilities.
    The targets are one-hot: exactly one outcome is 1.0.

    Args:
        result: the game outcome.

    Returns:
        Float32 array of length 6.
    """
    target = np.zeros(NUM_OUTPUTS, dtype=np.float32)
    mapping = {
        GameResult.WIN:             0,
        GameResult.WIN_GAMMON:      1,
        GameResult.WIN_BACKGAMMON:  2,
        GameResult.LOSE:            3,
        GameResult.LOSE_GAMMON:     4,
        GameResult.LOSE_BACKGAMMON: 5,
    }
    if result in mapping:
        target[mapping[result]] = 1.0
    return target


def td_lambda_update(
    network:   TDNetwork,
    optimizer: torch.optim.Optimizer,
    record:    GameRecord,
    td_lambda: float,
    device:    torch.device,
) -> float:
    """Perform TD(λ) weight update from a completed game.

    All states in the record are encoded from the initial player's
    perspective, and the result is also from the initial player's
    perspective. No perspective flipping needed.

    Batched: computes targets in one pass, then one forward+backward step.

    Args:
        network:   the network to update (should be on `device`).
        optimizer: the optimizer.
        record:    completed game record.
        td_lambda: the λ parameter (0 = TD(0), 1 = Monte Carlo).
        device:    torch device for training.

    Returns:
        Loss value for monitoring.
    """
    if len(record.states) < 2:
        return 0.0

    # Stack all states into a single batch
    states_np    = np.stack(record.states)
    states_batch = torch.from_numpy(states_np).to(device)

    # Phase 1: compute TD(λ) targets (no gradients)
    with torch.no_grad():
        all_predictions = network(states_batch)  # shape: (T, 6)

        # Final target from game outcome
        final_target = torch.from_numpy(
            result_to_target(record.result)
        ).to(device)

        # Build targets backward: target_t = (1-λ)*V(s_{t+1}) + λ*target_{t+1}
        T       = len(record.states)
        targets = torch.zeros_like(all_predictions)
        targets[-1] = final_target
        for t in range(T - 2, -1, -1):
            next_pred  = all_predictions[t + 1]
            targets[t] = (1 - td_lambda) * next_pred + td_lambda * targets[t + 1]

    # Phase 2: single forward pass with gradients + one optimizer step
    network.train()
    predictions = network(states_batch)
    loss        = nn.functional.mse_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_against(
    network:        TDNetwork,
    opponent:       TDNetwork,
    device:         torch.device,
    num_games:      int,
    variant:        Variant                      = Variant.STANDARD,
    rng:            Optional[np.random.Generator] = None,
    equity_weights: Optional[torch.Tensor]        = None,
) -> float:
    """Evaluate a network against an opponent.

    Both networks should be on `device` (CPU for evaluation).
    Plays num_games games, alternating who goes first, and returns
    the win rate of the primary network.

    Args:
        network:        the network to evaluate.
        opponent:       the opponent network.
        device:         torch device (CPU for evaluation).
        num_games:      number of games to play.
        variant:        game variant for starting positions.
        rng:            random number generator.
        equity_weights: optional equity weights for move selection.

    Returns:
        Win rate of `network` (0.0 to 1.0).
    """
    if rng is None:
        rng = np.random.default_rng()

    network.eval()
    opponent.eval()

    wins = 0
    for i in range(num_games):
        start = get_starting_position(variant)

        if i % 2 == 0:
            # network plays as initial player
            record = play_game(
                network, device,
                starting_state=start,
                opponent=opponent,
                epsilon=0.0, rng=rng,
                equity_weights=equity_weights,
            )
            if record.result.value > 0:
                wins += 1
        else:
            # opponent plays as initial player, network is second
            record = play_game(
                opponent, device,
                starting_state=start,
                opponent=network,
                epsilon=0.0, rng=rng,
                equity_weights=equity_weights,
            )
            if record.result.value < 0:
                wins += 1  # opponent lost = network wins

    return wins / num_games


class Trainer:
    """Main training orchestrator.

    Manages the self-play training loop, evaluation, SPRT promotion,
    and plateau detection.

    Device strategy:
    - play_device (CPU): used for self-play and evaluation games
    - train_device (GPU if available): used for TD(λ) backprop
    - Network moves between devices as needed
    - Frozen opponents always stay on play_device

    Args:
        config:     training configuration dictionary.
        device:     torch device for training (GPU if available).
        output_dir: directory for saving checkpoints and logs.
    """

    def __init__(
        self,
        config:     dict,
        device:     torch.device,
        output_dir: Path,
    ) -> None:
        self.config       = config
        self.train_device = device
        self.play_device  = torch.device("cpu")
        self.output_dir   = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Network — starts on CPU for initial self-play
        net_config    = config.get("network", {})
        self.network  = TDNetwork(
            input_size=BOARD_FEATURE_SIZE,
            hidden_layers=net_config.get("hidden_layers", [256, 256]),
            dropout=net_config.get("dropout", 0.0),
        ).to(self.play_device)

        # Optimizer
        train_config = config.get("training", {})
        self.lr      = train_config.get("learning_rate", 0.0001)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
        )

        # For small networks, CPU is faster than GPU due to transfer overhead.
        param_count = sum(p.numel() for p in self.network.parameters())
        if self.train_device.type == "cuda" and param_count < 1_000_000:
            print(f"Network has {param_count:,} parameters — using CPU for all computation")
            print(f"(GPU will be used when network exceeds 1M parameters)")
            self.train_device = self.play_device

        # TD parameters
        self.td_lambda           = train_config.get("td_lambda", 0.7)
        self.epsilon_start       = train_config.get("epsilon_start", 0.10)
        self.epsilon_end         = train_config.get("epsilon_end", 0.01)
        self.epsilon_decay_games = train_config.get("epsilon_decay_games", 200000)

        # Equity weights
        ew = train_config.get("equity_weights", [1, 2, 3, -1, -2, -3])
        self.equity_weights = torch.tensor(ew, dtype=torch.float32)
        print(f"Equity weights: {ew}")

        # Evaluation parameters
        eval_config          = config.get("evaluation", {})
        self.eval_cadence    = eval_config.get("cadence_games", 5000)
        self.eval_match_size = eval_config.get("eval_match_size", 100)

        # SPRT parameters
        sprt_config       = config.get("sprt", {})
        self.sprt_p0      = sprt_config.get("p0", 0.70)
        self.sprt_p1      = sprt_config.get("p1", 0.76)
        self.sprt_alpha   = sprt_config.get("alpha", 0.05)
        self.sprt_beta    = sprt_config.get("beta", 0.10)
        self.sprt_hard_cap = sprt_config.get("hard_cap", 2000)
        self.sprt_gate    = sprt_config.get("gate_threshold", 0.70)

        # Plateau parameters
        plateau_config                 = config.get("plateau", {})
        self.plateau_base_budget       = plateau_config.get("base_budget", 500000)
        self.plateau_budget_scale      = plateau_config.get("budget_scale", 1.5)
        self.staleness_window          = plateau_config.get("staleness_window", 200000)
        self.staleness_min_improvement = plateau_config.get("staleness_min_improvement", 0.02)
        self.max_failed_sprts          = plateau_config.get("max_failed_sprts", 3)

        # Variant for starting positions
        variant_str  = config.get("movegen", {}).get("variant", "standard")
        self.variant = Variant[variant_str.upper()]

        # Level tracking
        self.level_opponents: list[TDNetwork] = []
        self.stats = TrainingStats()
        self.rng   = np.random.default_rng(config.get("seed"))
        self._failed_sprts_this_level = 0

    def _to_play(self) -> None:
        """Move network to CPU for self-play/evaluation."""
        self.network.to(self.play_device)

    def _to_train(self) -> None:
        """Move network to GPU for backpropagation."""
        self.network.to(self.train_device)

    def _current_epsilon(self) -> float:
        """Compute current ε for ε-greedy exploration."""
        progress = min(
            self.stats.games_played / max(self.epsilon_decay_games, 1), 1.0
        )
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def _current_budget(self) -> int:
        """Compute the plateau budget for the current level."""
        level  = self.stats.current_level
        budget = int(self.plateau_base_budget * (self.plateau_budget_scale ** level))
        return budget

    def _check_staleness(self) -> bool:
        """Check if training has stalled.

        Returns True if improvement in rolling win rate over the
        staleness window is below the threshold.
        """
        history = self.stats.rolling_eval_history
        if len(history) < 2:
            return False

        window_evals = int(self.staleness_window / self.eval_cadence)
        if len(history) < window_evals:
            return False

        recent      = history[-1]
        old         = history[-window_evals]
        improvement = recent - old
        return improvement < self.staleness_min_improvement

    def _freeze_checkpoint(self) -> TDNetwork:
        """Create a frozen copy of the current network on CPU."""
        checkpoint = TDNetwork(
            input_size=BOARD_FEATURE_SIZE,
            hidden_layers=self.config.get("network", {}).get(
                "hidden_layers", [256, 256]
            ),
        ).to(self.play_device)
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        checkpoint.load_state_dict(state_dict)
        checkpoint.eval()
        return checkpoint

    def _save_checkpoint(self, label: str) -> Path:
        """Save the current network to disk."""
        path       = self.output_dir / f"checkpoint_{label}.pt"
        state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
        torch.save({
            "model_state_dict":     state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": {
                "games_played":   self.stats.games_played,
                "current_level":  self.stats.current_level,
                "levels_reached": self.stats.levels_reached,
            },
        }, path)
        return path

    def _run_sprt(self) -> bool:
        """Run the SPRT promotion test against the current level opponent.

        Network must be on CPU before calling.
        Returns True if the test accepts (promote).
        """
        if not self.level_opponents:
            return True

        opponent = self.level_opponents[-1]
        wins     = 0
        games    = 0

        self.network.eval()
        while True:
            start = get_starting_position(self.variant)

            if games % 2 == 0:
                record = play_game(
                    self.network, self.play_device,
                    starting_state=start,
                    opponent=opponent,
                    epsilon=0.0, rng=self.rng,
                    equity_weights=self.equity_weights,
                )
                if record.result.value > 0:
                    wins += 1
            else:
                record = play_game(
                    opponent, self.play_device,
                    starting_state=start,
                    opponent=self.network,
                    epsilon=0.0, rng=self.rng,
                    equity_weights=self.equity_weights,
                )
                if record.result.value < 0:
                    wins += 1

            games += 1

            result = sprt_test(
                wins, games,
                self.sprt_p0, self.sprt_p1,
                self.sprt_alpha, self.sprt_beta,
                self.sprt_hard_cap,
            )
            if result == SPRTResult.ACCEPT:
                self.stats.sprt_tests_passed += 1
                return True
            elif result == SPRTResult.REJECT:
                self.stats.sprt_tests_failed += 1
                return False

        return False

    def train(self, max_games: Optional[int] = None) -> TrainingStats:
        """Run the main training loop.

        Each iteration:
        1. Self-play one game on CPU
        2. Move network to GPU, run TD(λ) backprop
        3. Move network back to CPU

        Args:
            max_games: maximum total games to play (None = train until plateau).

        Returns:
            Final training statistics.
        """
        self.stats.training_start_time = time.time()

        print(f"Training: self-play on CPU, backprop on {self.train_device}")
        print(f"Network: {sum(p.numel() for p in self.network.parameters())} parameters")

        while True:
            # Termination conditions
            if max_games and self.stats.games_played >= max_games:
                print(f"Reached max games ({max_games})")
                break

            budget = self._current_budget()
            if self.stats.games_since_level_up >= budget:
                print(f"Plateau: budget exhausted ({budget} games) at level {self.stats.current_level}")
                break

            if self._check_staleness():
                print(f"Plateau: staleness detected at level {self.stats.current_level}")
                break

            # 1. Self-play on CPU
            self._to_play()
            epsilon = self._current_epsilon()
            start   = get_starting_position(
                self.variant,
                seed=int(self.rng.integers(0, 2**31)),
            )
            self.network.eval()
            record = play_game(
                self.network, self.play_device,
                starting_state=start,
                epsilon=epsilon, rng=self.rng,
                equity_weights=self.equity_weights,
            )

            # 2. TD(λ) update on train device
            self._to_train()
            loss = td_lambda_update(
                self.network, self.optimizer, record,
                self.td_lambda, self.train_device,
            )

            self.stats.games_played         += 1
            self.stats.games_since_level_up += 1
            self.stats.total_moves          += record.num_moves

            # 3. Periodic evaluation
            if self.stats.games_played % self.eval_cadence == 0:
                self._to_play()
                self._periodic_eval()

            # Progress reporting
            if self.stats.games_played % 1000 == 0:
                elapsed = time.time() - self.stats.training_start_time
                rate    = self.stats.games_played / elapsed
                print(
                    f"Games: {self.stats.games_played:,} | "
                    f"Level: {self.stats.current_level} | "
                    f"ε: {epsilon:.3f} | "
                    f"Win rate: {self.stats.rolling_win_rate:.3f} | "
                    f"Rate: {rate:.1f} games/s"
                )

        # Final save
        self._save_checkpoint(f"level{self.stats.current_level}_final")
        self.stats.levels_reached = self.stats.current_level

        elapsed = time.time() - self.stats.training_start_time
        print(f"\nTraining complete:")
        print(f"  Games: {self.stats.games_played:,}")
        print(f"  Levels reached: {self.stats.levels_reached}")
        print(f"  Time: {elapsed / 3600:.1f} hours")
        print(f"  Rate: {self.stats.games_played / elapsed:.1f} games/s")
        print(f"  SPRT tests: {self.stats.sprt_tests_run} "
              f"({self.stats.sprt_tests_passed} passed, "
              f"{self.stats.sprt_tests_failed} failed)")

        return self.stats

    def _periodic_eval(self) -> None:
        """Run periodic evaluation and check for promotion.

        Network must be on CPU before calling.
        """
        if not self.level_opponents:
            win_rate = self._eval_vs_random()
            self.stats.rolling_win_rate = win_rate
            self.stats.rolling_eval_history.append(win_rate)
            if win_rate > self.sprt_gate:
                print(f"  Level 0→1 gate reached (win rate vs random: {win_rate:.3f})")
                self._promote()
            return

        opponent = self.level_opponents[-1]
        win_rate = evaluate_against(
            self.network, opponent, self.play_device,
            self.eval_match_size, self.variant, self.rng,
            equity_weights=self.equity_weights,
        )
        self.stats.rolling_win_rate = win_rate
        self.stats.rolling_eval_history.append(win_rate)

        if win_rate >= self.sprt_gate:
            print(f"  SPRT gate reached (win rate: {win_rate:.3f}), running test...")
            self.stats.sprt_tests_run += 1
            if self._run_sprt():
                print(f"  ✓ SPRT passed! Promoting to level {self.stats.current_level + 1}")
                self._promote()
            else:
                self._failed_sprts_this_level += 1
                print(f"  ✗ SPRT failed ({self._failed_sprts_this_level}/{self.max_failed_sprts})")
                if self._failed_sprts_this_level >= self.max_failed_sprts:
                    remaining = self._current_budget() - self.stats.games_since_level_up
                    self.stats.games_since_level_up += remaining // 2
                    self._failed_sprts_this_level = 0  # reset after halving
                    print(f"  Budget halved after {self.max_failed_sprts} failed SPRTs")

    def _promote(self) -> None:
        """Promote to the next level."""
        checkpoint = self._freeze_checkpoint()
        self.level_opponents.append(checkpoint)
        self.stats.current_level        += 1
        self.stats.games_since_level_up  = 0
        self.stats.rolling_eval_history.clear()
        self.stats.rolling_win_rate      = 0.5
        self._failed_sprts_this_level    = 0
        self._save_checkpoint(f"level{self.stats.current_level}")
        print(f"  Saved level {self.stats.current_level} checkpoint")

    def _eval_vs_random(self) -> float:
        """Evaluate network against random play (level 0 baseline).

        Network must be on CPU before calling.
        """
        random_net = TDNetwork(input_size=BOARD_FEATURE_SIZE).to(self.play_device)
        random_net.eval()
        return evaluate_against(
            self.network, random_net, self.play_device,
            self.eval_match_size, self.variant, self.rng,
            equity_weights=self.equity_weights,
        )