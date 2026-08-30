"""TD-Gammon style neural network.

Feedforward network that takes a board state feature vector and outputs
six probabilities: P(win), P(win gammon), P(win backgammon),
P(lose), P(lose gammon), P(lose backgammon).

Architecture is configurable via hidden layer sizes. Uses ReLU activation
for hidden layers and sigmoid for the output layer.

Checkpoint architecture contract — a saved checkpoint carries its own
architecture, so nothing outside the file is needed to rebuild the
network (the same no-sidecar principle the ONNX `bgrl.*` metadata
contract applies to exported models; see `engine/export.py`):

    CHECKPOINT_ARCHITECTURE_KEY   checkpoint dict key holding the mapping
                                  produced by `TDNetwork.architecture`
                                  ({"input_size", "hidden_layers"}).

`TDNetwork.from_state_dict` prefers that embedded architecture and
verifies it against the weight shapes, refusing a checkpoint whose
self-description and weights disagree. Checkpoints written before the
contract existed carry no such key; they still load, by inferring the
architecture from the weight shapes.
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

# Checkpoint dict key under which the trainer embeds TDNetwork.architecture.
# Absent from checkpoints saved before the contract existed; those load by
# inferring the architecture from the weight shapes.
CHECKPOINT_ARCHITECTURE_KEY = "network_architecture"


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

        self._input_size = input_size
        self._hidden_layers = list(hidden_layers)

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

    @property
    def input_size(self) -> int:
        """Size of the input feature vector."""
        return self._input_size

    @property
    def hidden_layers(self) -> list[int]:
        """Hidden layer sizes (copy; the architecture is immutable)."""
        return list(self._hidden_layers)

    @property
    def architecture(self) -> dict:
        """Self-description of this network's architecture.

        The keys are the constructor parameters that determine the weight
        shapes, so ``TDNetwork(**net.architecture)`` rebuilds the same
        shape. Dropout is deliberately excluded: it leaves no trace in the
        weights and is irrelevant for inference.

        This mapping is what the trainer embeds in a checkpoint under
        `CHECKPOINT_ARCHITECTURE_KEY`.
        """
        return {
            "input_size": self._input_size,
            "hidden_layers": list(self._hidden_layers),
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        architecture: dict | None = None,
    ) -> TDNetwork:
        """Reconstruct a TDNetwork from a saved state dict.

        The architecture comes from `architecture` when a checkpoint
        embeds one (see `CHECKPOINT_ARCHITECTURE_KEY`); it is otherwise
        inferred from the Linear weight shapes, so checkpoints written
        before the contract existed still load with no accompanying
        config. An embedded architecture is always cross-checked against
        the weight shapes: disagreement means the checkpoint is corrupt
        or hand-edited, and neither side is trusted over the other.

        Dropout is not represented in a state dict and is irrelevant for
        inference; the reconstructed network has dropout disabled.

        Args:
            state_dict: a TDNetwork state dict (as saved by torch.save).
            architecture: the checkpoint's embedded self-description, or
                          None to infer the architecture from the weights.

        Returns:
            A TDNetwork with the resolved architecture and loaded weights.

        Raises:
            ValueError: state dict does not describe a TDNetwork
                        (no Linear weights, mismatched chain, or wrong
                        output size), the embedded architecture is
                        malformed, or it disagrees with the weight shapes.
        """
        inferred = cls._infer_architecture(state_dict)

        if architecture is None:
            resolved = inferred
        else:
            resolved = cls._coerce_architecture(architecture)
            if resolved != inferred:
                raise ValueError(
                    "checkpoint architecture disagrees with its weight "
                    f"shapes: embedded {resolved}, inferred {inferred}"
                )

        network = cls(
            input_size=resolved["input_size"],
            hidden_layers=resolved["hidden_layers"],
        )
        network.load_state_dict(state_dict)
        return network

    @staticmethod
    def _infer_architecture(state_dict: dict) -> dict:
        """Infer the architecture from a state dict's Linear weight shapes.

        Raises:
            ValueError: the state dict does not describe a TDNetwork.
        """
        # Linear weights are the 2-D entries, named "network.{i}.weight";
        # sort by module index to guard against reordered dicts.
        weight_keys = sorted(
            (k for k, v in state_dict.items()
             if k.endswith(".weight") and v.dim() == 2),
            key=lambda k: int(k.split(".")[1]),
        )
        if not weight_keys:
            raise ValueError("state dict contains no Linear weights")

        shapes = [tuple(state_dict[k].shape) for k in weight_keys]
        for (out_prev, _), (_, in_next) in zip(shapes, shapes[1:]):
            if out_prev != in_next:
                raise ValueError(
                    f"state dict layer shapes do not chain: {shapes}"
                )
        if shapes[-1][0] != NUM_OUTPUTS:
            raise ValueError(
                f"state dict output size is {shapes[-1][0]}, "
                f"expected {NUM_OUTPUTS}"
            )

        return {
            "input_size": shapes[0][1],
            "hidden_layers": [out for out, _ in shapes[:-1]],
        }

    @staticmethod
    def _coerce_architecture(architecture: dict) -> dict:
        """Normalize an embedded architecture to the canonical form.

        Raises:
            ValueError: the mapping lacks the required keys or carries
                        values that are not sizes.
        """
        try:
            resolved = {
                "input_size": int(architecture["input_size"]),
                "hidden_layers": [
                    int(size) for size in architecture["hidden_layers"]
                ],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"embedded architecture is malformed: {architecture!r}"
            ) from exc
        return resolved

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


def compute_equity(
    output: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute equity (expected value) from network output.

    The weights determine how each outcome contributes to equity.
    Common configurations:
        Money play:      [1, 2, 3, -1, -2, -3]
        DMP (1a1a):      [1, 1, 1, -1, -1, -1]
        Leader 1a2aC:    [1, 1, 1, -1, -2, -3]
        Trailer 2a1aC:   [1, 2, 3, -1, -1, -1]

    Args:
        output: network output tensor of shape (..., 6).
        weights: equity weights tensor of shape (6,).
                 Defaults to money play weights.

    Returns:
        Equity tensor of shape (...,).
    """
    if weights is None:
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
