"""BgRLEngine — export a trainer checkpoint to ONNX.

Usage:
    python export_onnx.py CHECKPOINT [--output PATH] [--model-role ROLE]

Exported models are deployment artifacts, not source: they are produced
locally on demand and are never committed (the committed parity fixture
under parity/ is the only .onnx in version control). Consumers identify
a model durably via its embedded bgrl.checkpoint_sha256 provenance and
verify compatibility via bgrl.encoding_version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from engine.export import export_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a BgRLEngine checkpoint to ONNX"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a trainer checkpoint (.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path (default: checkpoint path with .onnx suffix)",
    )
    parser.add_argument(
        "--model-role",
        type=str,
        default="general",
        help="Routing role stamped as bgrl.model_role (default: general)",
    )
    args = parser.parse_args()

    output_path, metadata = export_checkpoint(
        args.checkpoint, args.output, model_role=args.model_role
    )

    print(f"Exported {args.checkpoint} -> {output_path}")
    for key, value in metadata.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
