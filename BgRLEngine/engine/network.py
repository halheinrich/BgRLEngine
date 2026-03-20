"""TD-Gammon style neural network.

Feedforward network that takes a board state feature vector and outputs
six probabilities: P(win), P(win gammon), P(win backgammon),
P(lose), P(lose gammon), P(lose backgammon).

Architecture is configurable via hidden layer sizes. Uses ReLU activation
for hidden layers and sigmoid for the output layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from engine.state import BOARD_FEATURE_SIZE, FULL_FEATURE_SIZE

# Output indices
OUT_WIN = 0
OUT_WIN_GAMMON = 1
OUT_WIN_BG = 2
OUT_LOSE = 3
OUT_LOSE_GAMMON = 4
OUT_LOSE_BG = 5
NUM_OUTPUTS = 6


class TDNetwork(nn.Module):
    """TD-Gammon feedforward neural network.

    Args:
        input_size: size of input feature vector.
        hidden_layers: list of hidden layer sizes (e.g., [256, 256]).
        dropout: dropout probability (0 = disabled).
    """

    def __init__(
        self,
        input_size: int = BOARD_FEATURE_SIZE,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256]

        layers: list[nn.Module] = []
        prev_size = input_size

        for h_size in hidden_layers:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = h_size

        layers.append(nn.Linear(prev_size, NUM_OUTPUTS))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier uniform for better training start
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (batch, input_size) or (input_size,).

        Returns:
            Tensor of shape (batch, 6) or (6,) with output probabilities.
        """
        return self.network(x)

    def evaluate(self, features: torch.Tensor) -> torch.Tensor:
        """Evaluate a position (no gradient tracking).

        Convenience method for inference. Handles adding batch dimension
        if needed.

        Args:
            features: feature tensor, shape (input_size,) or (batch, input_size).

        Returns:
            Output tensor with probabilities.
        """
        with torch.no_grad():
            if features.dim() == 1:
                features = features.unsqueeze(0)
            return self.forward(features).squeeze(0)


def compute_equity(output: torch.Tensor) -> torch.Tensor:
    """Compute equity (expected value) from network output.

    Equity = P(win) + P(win_gammon) + P(win_bg) - P(lose) - P(lose_gammon) - P(lose_bg)

    For money play, gammons are worth 2x and backgammons 3x:
    Equity = P(win)*(1) + P(win_gammon)*(2) + P(win_bg)*(3)
           - P(lose)*(1) - P(lose_gammon)*(2) - P(lose_bg)*(3)

    Note: the output probabilities represent the probability of each
    outcome, not cumulative. So P(win) is the probability of a plain
    win (not gammon, not backgammon).

    Args:
        output: network output tensor of shape (..., 6).

    Returns:
        Equity tensor of shape (...,).
    """
    # Money play equity with gammon/backgammon values
    weights = torch.tensor(
        [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
        device=output.device,
        dtype=output.dtype,
    )
    return (output * weights).sum(dim=-1)


def compute_match_equity(
    output: torch.Tensor,
    gammon_value_win: float,
    gammon_value_lose: float,
    bg_value_win: float = 0.0,
    bg_value_lose: float = 0.0,
) -> torch.Tensor:
    """Compute match equity with score-dependent gammon values.

    The gammon values come from the match equity table (MET) and
    represent how much more a gammon win/loss is worth compared
    to a plain win/loss at the current score.

    Args:
        output: network output tensor of shape (..., 6).
        gammon_value_win: extra value of winning a gammon vs plain win.
        gammon_value_lose: extra cost of losing a gammon vs plain loss.
        bg_value_win: extra value of winning a backgammon.
        bg_value_lose: extra cost of losing a backgammon.

    Returns:
        Match equity tensor of shape (...,).
    """
    weights = torch.tensor(
        [
            1.0,
            1.0 + gammon_value_win,
            1.0 + bg_value_win,
            -1.0,
            -(1.0 + gammon_value_lose),
            -(1.0 + bg_value_lose),
        ],
        device=output.device,
        dtype=output.dtype,
    )
    return (output * weights).sum(dim=-1)
