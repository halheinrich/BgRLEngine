"""ONNX export of TDNetwork models.

This module is the producer side of the cross-language model contract
consumed by BgInference (C# / ONNX Runtime).

Graph contract:
    input  "features"       float32 [batch, input_size]   (dynamic batch)
    output "probabilities"  float32 [batch, 6]

Metadata contract — stamped into the .onnx file's metadata_props (no
sidecar: the contract travels inside the model file; ONNX Runtime exposes
it as ModelMetadata.CustomMetadataMap). All values are strings per the
ONNX spec. Keys always present:

    bgrl.encoding_version   board→feature encoding version
                            (engine.state.ENCODING_VERSION). The consumer
                            holds its own required version and must hard-
                            fail on mismatch.
    bgrl.input_size         input feature count (e.g. "303")
    bgrl.num_outputs        output count ("6")
    bgrl.output_semantics   comma-separated output component names
    bgrl.model_role         which engine this model is, e.g. "general"
                            ("parity" for the synthetic parity fixture)
    bgrl.hidden_layers      JSON list of hidden layer sizes (provenance)

Checkpoint exports additionally carry provenance keys:

    bgrl.checkpoint_file, bgrl.checkpoint_sha256,
    bgrl.checkpoint_games_played, bgrl.checkpoint_level,
    bgrl.torch_version, bgrl.export_timestamp
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import onnx
import torch

from engine.network import NUM_OUTPUTS, TDNetwork
from engine.state import ENCODING_VERSION

# Opset 17: stable, supported by all current ONNX Runtime releases, and
# fully covers this graph (Gemm / Relu / Sigmoid).
ONNX_OPSET = 17

INPUT_NAME = "features"
OUTPUT_NAME = "probabilities"

METADATA_KEY_PREFIX = "bgrl."

OUTPUT_SEMANTICS = (
    "p_win,p_win_gammon,p_win_backgammon,"
    "p_lose,p_lose_gammon,p_lose_backgammon"
)


def export_network(
    network: TDNetwork,
    output_path: str | Path,
    *,
    model_role: str,
    provenance: dict[str, str] | None = None,
) -> dict[str, str]:
    """Export a TDNetwork to an ONNX file with the bgrl.* metadata contract.

    The network is exported in eval mode on CPU with a dynamic batch
    dimension. The file is verified with the ONNX checker before this
    function returns.

    Args:
        network: the network to export.
        output_path: destination .onnx path.
        model_role: routing role stamped as bgrl.model_role
                    (e.g. "general"; "parity" for the parity fixture).
        provenance: optional extra metadata entries; keys are stamped
                    with the bgrl. prefix added.

    Returns:
        The complete metadata mapping stamped into the file
        (fully prefixed keys).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    network = network.cpu().eval()
    dummy_input = torch.zeros(1, network.input_size, dtype=torch.float32)

    torch.onnx.export(
        network,
        (dummy_input,),
        str(output_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch"},
            OUTPUT_NAME: {0: "batch"},
        },
        opset_version=ONNX_OPSET,
    )

    metadata = {
        "encoding_version": str(ENCODING_VERSION),
        "input_size": str(network.input_size),
        "num_outputs": str(NUM_OUTPUTS),
        "output_semantics": OUTPUT_SEMANTICS,
        "model_role": model_role,
        "hidden_layers": json.dumps(network.hidden_layers),
    }
    if provenance:
        metadata.update(provenance)
    metadata = {METADATA_KEY_PREFIX + k: v for k, v in metadata.items()}

    model = onnx.load(str(output_path))
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    return metadata


def export_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    *,
    model_role: str = "general",
) -> tuple[Path, dict[str, str]]:
    """Export a trainer checkpoint (.pt) to ONNX with full provenance.

    The architecture is inferred from the checkpoint's model state dict
    (TDNetwork.from_state_dict), so no config file is needed.

    Args:
        checkpoint_path: path to a checkpoint saved by the trainer.
        output_path: destination .onnx path; defaults to the checkpoint
                     path with an .onnx suffix.
        model_role: routing role stamped as bgrl.model_role.

    Returns:
        (output path, stamped metadata mapping).
    """
    checkpoint_path = Path(checkpoint_path)
    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    network = TDNetwork.from_state_dict(checkpoint["model_state_dict"])

    stats = checkpoint.get("stats", {})
    provenance = {
        "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
        "checkpoint_games_played": str(stats.get("games_played", "unknown")),
        "checkpoint_level": str(stats.get("current_level", "unknown")),
        "torch_version": torch.__version__,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metadata = export_network(
        network, output_path, model_role=model_role, provenance=provenance
    )
    return Path(output_path), metadata
