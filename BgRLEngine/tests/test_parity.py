"""Producer-side parity tests: PyTorch vs ONNX Runtime.

These tests catch export drift at the producer, before the C# consumer
(BgInference) ever sees a model:

    1. Export round-trip — a freshly exported network produces the same
       outputs under ONNX Runtime as under PyTorch.
    2. Metadata contract — every export carries the bgrl.* handshake keys
       the consumer fail-fasts on.
    3. Committed parity fixtures (parity/model.onnx + parity/vectors.json)
       — the executable cross-language contract. These must never skip:
       a fresh checkout that silently lacked them would let the C# parity
       gate no-op, which is the failure mode the fixtures exist to prevent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from engine.export import (
    INPUT_NAME,
    ONNX_OPSET,
    OUTPUT_NAME,
    OUTPUT_SEMANTICS,
    export_checkpoint,
    export_network,
)
from engine.network import (
    CHECKPOINT_ARCHITECTURE_KEY,
    NUM_OUTPUTS,
    TDNetwork,
)
from engine.state import (
    BOARD_FEATURE_SIZE,
    ENCODING_VERSION,
    BoardState,
    encode_board,
    encode_board_batch,
)

# PyTorch and ONNX Runtime use different CPU kernels for the same float32
# graph; tiny last-ulp differences are expected, real drift is not.
ROUND_TRIP_ATOL = 1e-6


def ort_session(path: str | Path) -> ort.InferenceSession:
    return ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )


def ort_forward(session: ort.InferenceSession, features: np.ndarray) -> np.ndarray:
    return session.run([OUTPUT_NAME], {INPUT_NAME: features})[0]


@pytest.fixture()
def exported_model(tmp_path):
    """A freshly exported small network and its ONNX path."""
    torch.manual_seed(20260702)
    network = TDNetwork(hidden_layers=[32, 32]).eval()
    path = tmp_path / "model.onnx"
    metadata = export_network(network, path, model_role="general")
    return network, path, metadata


class TestExportRoundTrip:
    def test_fresh_network_matches_onnxruntime(self, exported_model):
        network, path, _ = exported_model
        rng = np.random.default_rng(0)
        features = rng.random((16, BOARD_FEATURE_SIZE), dtype=np.float32)

        with torch.no_grad():
            expected = network(torch.from_numpy(features)).numpy()
        actual = ort_forward(ort_session(path), features)

        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=ROUND_TRIP_ATOL
        )

    def test_dynamic_batch(self, exported_model):
        _, path, _ = exported_model
        session = ort_session(path)
        rng = np.random.default_rng(1)
        for batch in (1, 7, 100):
            features = rng.random(
                (batch, BOARD_FEATURE_SIZE), dtype=np.float32
            )
            assert ort_forward(session, features).shape == (batch, NUM_OUTPUTS)

    def test_metadata_contract(self, exported_model):
        _, path, stamped = exported_model
        # Read back through ONNX Runtime — the consumer's view.
        metadata = ort_session(path).get_modelmeta().custom_metadata_map

        assert metadata["bgrl.encoding_version"] == str(ENCODING_VERSION)
        assert metadata["bgrl.input_size"] == str(BOARD_FEATURE_SIZE)
        assert metadata["bgrl.num_outputs"] == str(NUM_OUTPUTS)
        assert metadata["bgrl.output_semantics"] == OUTPUT_SEMANTICS
        assert metadata["bgrl.model_role"] == "general"
        assert json.loads(metadata["bgrl.hidden_layers"]) == [32, 32]
        assert metadata == stamped


class TestCheckpointExport:
    """Round-trip a real trainer checkpoint (local artifact, gitignored).

    Skips loudly when no local checkpoint exists — the committed parity
    fixtures, not this test, are the gate that must never skip.
    """

    CHECKPOINT = Path("output/checkpoint_level4.pt")

    def test_checkpoint_round_trip(self, tmp_path):
        if not self.CHECKPOINT.exists():
            pytest.skip(
                f"local training checkpoint {self.CHECKPOINT} not present "
                f"(output/ is gitignored); run training or point CHECKPOINT "
                f"at another .pt to exercise the real-checkpoint export path"
            )

        path, metadata = export_checkpoint(
            self.CHECKPOINT, tmp_path / "checkpoint.onnx"
        )

        checkpoint = torch.load(
            self.CHECKPOINT, map_location="cpu", weights_only=True
        )
        network = TDNetwork.from_state_dict(
            checkpoint["model_state_dict"],
            architecture=checkpoint.get(CHECKPOINT_ARCHITECTURE_KEY),
        ).eval()

        rng = np.random.default_rng(2)
        features = rng.random((32, BOARD_FEATURE_SIZE), dtype=np.float32)
        with torch.no_grad():
            expected = network(torch.from_numpy(features)).numpy()
        actual = ort_forward(ort_session(path), features)

        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=ROUND_TRIP_ATOL
        )
        assert "bgrl.checkpoint_sha256" in metadata
        assert metadata["bgrl.checkpoint_file"] == self.CHECKPOINT.name


# ── Committed parity fixtures — the cross-language contract ────────

PARITY_MODEL = Path("parity/model.onnx")
PARITY_VECTORS = Path("parity/vectors.json")


def board_from_case(case: dict) -> BoardState:
    """Rebuild a BoardState from a vectors.json board record."""
    board = case["board"]
    state = BoardState()
    state.points = np.array(board["points"], dtype=np.int16)
    state.bar_player = board["bar_player"]
    state.bar_opponent = board["bar_opponent"]
    state.off_player = board["off_player"]
    state.off_opponent = board["off_opponent"]
    state.player_to_move = board["player_to_move"]
    return state


@pytest.fixture(scope="module")
def vectors() -> dict:
    # No skip on absence: a missing fixture is a broken checkout and
    # must fail the suite loudly.
    assert PARITY_MODEL.exists(), f"committed fixture missing: {PARITY_MODEL}"
    assert PARITY_VECTORS.exists(), f"committed fixture missing: {PARITY_VECTORS}"
    with open(PARITY_VECTORS, encoding="utf-8") as f:
        return json.load(f)


class TestCommittedParityFixtures:
    def test_model_hash_pairs_with_vectors(self, vectors):
        actual = hashlib.sha256(PARITY_MODEL.read_bytes()).hexdigest()
        assert actual == vectors["model_sha256"], (
            "parity/model.onnx does not match vectors.json's model_sha256 — "
            "the fixture pair is inconsistent; regenerate both with "
            "python -m parity.generate_vectors"
        )

    def test_metadata_handshake(self, vectors):
        metadata = ort_session(PARITY_MODEL).get_modelmeta().custom_metadata_map

        assert vectors["encoding_version"] == ENCODING_VERSION
        assert metadata["bgrl.encoding_version"] == str(ENCODING_VERSION)
        assert metadata["bgrl.input_size"] == str(vectors["input_size"])
        assert metadata["bgrl.num_outputs"] == str(vectors["num_outputs"])
        assert metadata["bgrl.output_semantics"] == vectors["output_semantics"]
        assert metadata["bgrl.model_role"] == "parity"
        # The committed binary must be reproducible: no timestamp.
        assert "bgrl.export_timestamp" not in metadata

    def test_encoding_reproduces_features_bit_exact(self, vectors):
        states = [board_from_case(c) for c in vectors["cases"]]
        expected = np.array(
            [c["features"] for c in vectors["cases"]], dtype=np.float32
        )

        scalar = np.stack([encode_board(s) for s in states])
        batch = encode_board_batch(states)

        assert np.array_equal(scalar, expected), (
            "encode_board no longer reproduces the committed parity "
            "features — if the encoding change is intentional, bump "
            "ENCODING_VERSION and regenerate the fixtures"
        )
        assert np.array_equal(batch, expected)

    def test_onnxruntime_reproduces_outputs(self, vectors):
        features = np.array(
            [c["features"] for c in vectors["cases"]], dtype=np.float32
        )
        expected = np.array(
            [c["output"] for c in vectors["cases"]], dtype=np.float32
        )
        actual = ort_forward(ort_session(PARITY_MODEL), features)
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=vectors["output_tolerance_abs"]
        )

    def test_case_coverage(self, vectors):
        """Guard against a degenerate future regeneration of the fixture."""
        cases = vectors["cases"]
        boards = [c["board"] for c in cases]

        assert len(cases) >= 20
        assert len({c["label"] for c in cases}) == len(cases)
        assert any(b["bar_player"] > 0 for b in boards)
        assert any(b["off_player"] >= 15 for b in boards)  # game over
        assert any(not b["player_to_move"] for b in boards)
        assert any(max(b["points"]) > 5 for b in boards)  # overflow
        assert any(  # empty board: the total_pips == 0 branch
            all(p == 0 for p in b["points"]) and b["bar_player"] == 0
            for b in boards
        )
