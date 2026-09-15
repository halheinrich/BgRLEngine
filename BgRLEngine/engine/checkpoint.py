"""Training checkpoint format: the one owner of the .pt file's shape.

A checkpoint is the trainer's durable output and the input every
downstream tool starts from (ONNX export, config comparison). This
module writes it (`save_checkpoint`) and reads it (`load_checkpoint`).
The dict inside the file is this module's and nobody else's: no caller
builds it or indexes it, so the format changes in exactly one place.

Checkpoint contract — every save stamps two self-description keys beside
the weights, the optimizer state and the trainer stats (the same
no-sidecar principle the ONNX `bgrl.*` metadata contract applies to
exported models; see `engine/export.py`):

    CHECKPOINT_ARCHITECTURE_KEY       the mapping produced by
                                      `TDNetwork.architecture`
                                      ({"input_size", "hidden_layers"}),
                                      so nothing outside the file is
                                      needed to rebuild the network.
    CHECKPOINT_ENCODING_VERSION_KEY   `engine.state.ENCODING_VERSION` at
                                      save time: the board→feature
                                      encoding the weights were trained
                                      against.

`load_checkpoint` enforces both:

    Encoding handshake. A checkpoint stamped with an encoding version
    other than the current `ENCODING_VERSION` refuses to load, naming
    both versions: weights trained against one encoding evaluate
    silently wrong under another of the same input size, and nothing
    migrates on load. A checkpoint with no stamp predates the handshake
    and is read as encoding 1, with a warning: encoding 1 is the only
    encoding that existed before the stamp, and every save now stamps,
    so the unstamped case dies out. After an encoding bump an unstamped
    checkpoint is still read as encoding 1, and so refuses.

    Architecture. The embedded architecture is preferred and
    cross-checked against the weight shapes by
    `TDNetwork.from_state_dict`, which refuses a checkpoint whose
    self-description and weights disagree. A checkpoint written before
    the architecture key existed loads by inferring the architecture
    from the weight shapes; that fallback is permanent, not a migration
    window (`output/` is gitignored, so such checkpoints are exactly the
    ones that cannot be regenerated).
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from engine.network import TDNetwork
from engine.state import ENCODING_VERSION

# Self-description keys stamped by every save. Importable so the contract
# tests can inspect a saved file by key; no reader or writer outside this
# module uses them.
CHECKPOINT_ARCHITECTURE_KEY = "network_architecture"
CHECKPOINT_ENCODING_VERSION_KEY = "encoding_version"

# The remaining keys are private to the format.
_MODEL_STATE_KEY = "model_state_dict"
_OPTIMIZER_STATE_KEY = "optimizer_state_dict"
_STATS_KEY = "stats"

# The encoding an unstamped checkpoint is read as: the only encoding that
# existed before the stamp.
_UNSTAMPED_ENCODING_VERSION = 1


@dataclass(frozen=True)
class CheckpointStats:
    """Trainer progress recorded in a checkpoint.

    Attributes:
        games_played: self-play games completed at save time.
        current_level: curriculum level being trained at save time.
        levels_reached: curriculum levels promoted through.
    """

    games_played: int
    current_level: int
    levels_reached: int


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A checkpoint that has passed the handshake.

    Attributes:
        network: the saved network, rebuilt from the embedded (or, for a
                 pre-contract checkpoint, inferred) architecture. Dropout
                 is disabled; the module is in its default training mode.
        stats: the trainer progress recorded at save time.
    """

    network: TDNetwork
    stats: CheckpointStats


def save_checkpoint(
    path: str | Path,
    network: TDNetwork,
    optimizer: torch.optim.Optimizer,
    stats: CheckpointStats,
) -> None:
    """Write a checkpoint stamped with both self-description keys.

    The weights are moved to CPU so the file loads on any device.

    Args:
        path: destination .pt path.
        network: the network whose weights and architecture are saved.
        optimizer: the optimizer whose state is saved beside the weights.
        stats: the trainer progress to record.
    """
    state_dict = {k: v.cpu() for k, v in network.state_dict().items()}
    torch.save({
        _MODEL_STATE_KEY:                state_dict,
        _OPTIMIZER_STATE_KEY:            optimizer.state_dict(),
        CHECKPOINT_ARCHITECTURE_KEY:     network.architecture,
        CHECKPOINT_ENCODING_VERSION_KEY: ENCODING_VERSION,
        _STATS_KEY:                      dataclasses.asdict(stats),
    }, path)


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Read a checkpoint, enforcing the encoding handshake.

    The one path by which a checkpoint becomes a network: the handshake
    runs before any weights are trusted, then the network is rebuilt via
    `TDNetwork.from_state_dict` with the embedded architecture.

    Args:
        path: a checkpoint written by `save_checkpoint` (or by the trainer
              before this module existed).

    Returns:
        The rebuilt network and the recorded stats.

    Raises:
        ValueError: the checkpoint was trained under another encoding
                    version (the message names both), its encoding stamp
                    or stats are malformed, or its embedded architecture
                    disagrees with its weight shapes.

    Warns:
        UserWarning: the checkpoint carries no encoding stamp and is read
                     as encoding 1.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _check_encoding_version(checkpoint, path)
    network = TDNetwork.from_state_dict(
        checkpoint[_MODEL_STATE_KEY],
        architecture=checkpoint.get(CHECKPOINT_ARCHITECTURE_KEY),
    )
    return LoadedCheckpoint(network=network, stats=_read_stats(checkpoint, path))


def _check_encoding_version(checkpoint: dict, path: Path) -> None:
    """Refuse a checkpoint trained under another encoding.

    Raises:
        ValueError: the stamp is malformed, or the stamped (or assumed)
                    version is not the current `ENCODING_VERSION`.
    """
    stamped = CHECKPOINT_ENCODING_VERSION_KEY in checkpoint
    if stamped:
        version = checkpoint[CHECKPOINT_ENCODING_VERSION_KEY]
        # bool is an int subclass; a True stamp is corruption, not version 1.
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(
                f"checkpoint {path} carries a malformed encoding version "
                f"stamp: {version!r}"
            )
    else:
        version = _UNSTAMPED_ENCODING_VERSION
        # stacklevel 3: report the caller of load_checkpoint, not this helper.
        warnings.warn(
            f"checkpoint {path} carries no encoding version stamp; "
            f"assuming encoding version {version}, the only encoding that "
            f"predates the stamp",
            stacklevel=3,
        )

    if version != ENCODING_VERSION:
        assumed = "" if stamped else " (assumed: the file is unstamped)"
        raise ValueError(
            f"checkpoint {path} was trained under encoding version "
            f"{version}{assumed}, but this engine uses encoding version "
            f"{ENCODING_VERSION}; a checkpoint does not migrate across "
            f"encodings"
        )


def _read_stats(checkpoint: dict, path: Path) -> CheckpointStats:
    """Read the recorded trainer progress.

    Raises:
        ValueError: the stats are missing a field or carry a non-integer.
    """
    raw = checkpoint.get(_STATS_KEY)
    try:
        values = {
            field.name: raw[field.name]
            for field in dataclasses.fields(CheckpointStats)
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"checkpoint {path} carries malformed stats: {raw!r}"
        ) from exc
    if not all(
        isinstance(v, int) and not isinstance(v, bool) for v in values.values()
    ):
        raise ValueError(f"checkpoint {path} carries malformed stats: {raw!r}")
    return CheckpointStats(**values)
