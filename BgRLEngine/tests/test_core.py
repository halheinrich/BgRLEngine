"""Tests for BgRLEngine core modules."""

import numpy as np
import pytest
import yaml
from pathlib import Path

from engine.state import (
    BoardState, encode_board, encode_point, encode_bar,
    encode_borne_off, flip_perspective,
    BOARD_FEATURE_SIZE, NUM_POINTS, UNITS_PER_POINT,
)
from engine.dice import roll_dice, generate_plays, get_dice_to_use
from engine.network import TDNetwork, compute_equity, NUM_OUTPUTS
import engine.checkpoint
from engine.checkpoint import (
    CHECKPOINT_ARCHITECTURE_KEY, CHECKPOINT_ENCODING_VERSION_KEY,
    CheckpointStats, load_checkpoint,
)
from training.td_trainer import (
    Trainer, sprt_test, SPRTResult, result_to_target,
)


# ── BgMoveGen fixture ──────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def load_movegen_fixture():
    """Load BgMoveGen DLL once per test session."""
    from engine.movegen import load_movegen, Variant
    config_path = Path("configs/default.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    dll_path = config["movegen"]["dll_path"]
    load_movegen(dll_path)


# ── State encoding tests ───────────────────────────────────────────

class TestEncodePoint:
    def test_zero_checkers(self):
        f = encode_point(0)
        assert len(f) == UNITS_PER_POINT
        assert all(f == 0)

    def test_one_checker(self):
        f = encode_point(1)
        assert f[0] == 1.0
        assert all(f[1:] == 0)

    def test_five_checkers(self):
        f = encode_point(5)
        assert all(f[:5] == 1.0)
        assert f[5] == 0.0

    def test_eight_checkers(self):
        f = encode_point(8)
        assert all(f[:5] == 1.0)
        assert f[5] == pytest.approx(3.0 / 10.0)

    def test_fifteen_checkers(self):
        f = encode_point(15)
        assert all(f[:5] == 1.0)
        assert f[5] == pytest.approx(1.0)  # capped at 1.0


class TestEncodeBar:
    def test_zero(self):
        assert all(encode_bar(0) == 0)

    def test_one(self):
        f = encode_bar(1)
        assert f[0] == 1.0 and f[1] == 0.0 and f[2] == 0.0

    def test_three_plus(self):
        f = encode_bar(5)
        assert all(f == 1.0)


class TestEncodeBorneOff:
    def test_none(self):
        f = encode_borne_off(0)
        assert f[0] == 0.0 and f[1] == 0.0

    def test_some(self):
        f = encode_borne_off(7)
        assert f[0] == pytest.approx(7 / 15)
        assert f[1] == 0.0

    def test_all(self):
        f = encode_borne_off(15)
        assert f[0] == pytest.approx(1.0)
        assert f[1] == 1.0


class TestBoardState:
    def test_standard_setup_checker_count(self):
        state = BoardState.standard_setup()
        player = sum(max(0, state.points[i]) for i in range(24))
        opponent = sum(abs(min(0, state.points[i])) for i in range(24))
        assert player == 15
        assert opponent == 15

    def test_nackgammon_setup_checker_count(self):
        from engine.movegen import get_starting_position, Variant
        state = get_starting_position(Variant.NACKGAMMON)
        player = sum(max(0, state.points[i]) for i in range(24))
        opponent = sum(abs(min(0, state.points[i])) for i in range(24))
        assert player == 15
        assert opponent == 15

    def test_nackgammon_layout(self):
        """Verify authoritative Nackgammon position."""
        from engine.movegen import get_starting_position, Variant
        state = get_starting_position(Variant.NACKGAMMON)
        # Player: 4@idx5, 3@idx7, 4@idx12, 2@idx22, 2@idx23
        assert state.points[5]  ==  4
        assert state.points[7]  ==  3
        assert state.points[12] ==  4
        assert state.points[22] ==  2
        assert state.points[23] ==  2
        # Opponent: -4@idx18, -3@idx16, -4@idx11, -2@idx1, -2@idx0
        assert state.points[18] == -4
        assert state.points[16] == -3
        assert state.points[11] == -4
        assert state.points[1]  == -2
        assert state.points[0]  == -2

    def test_standard_pip_count(self):
        state = BoardState.standard_setup()
        assert state.player_pip_count() == 167
        assert state.opponent_pip_count() == 167

    def test_is_race_standard(self):
        state = BoardState.standard_setup()
        assert not state.is_race()

    def test_is_race_separated(self):
        state = BoardState()
        state.points[0] = 5
        state.points[3] = 5
        state.points[5] = 5
        state.points[18] = -5
        state.points[20] = -5
        state.points[23] = -5
        assert state.is_race()

    def test_encode_board_size(self):
        state = BoardState.standard_setup()
        features = encode_board(state)
        assert len(features) == BOARD_FEATURE_SIZE

    def test_encode_board_dtype(self):
        state = BoardState.standard_setup()
        features = encode_board(state)
        assert features.dtype == np.float32


class TestFlipPerspective:
    def test_flip_preserves_checkers(self):
        state = BoardState.standard_setup()
        flipped = flip_perspective(state)
        player_orig = sum(max(0, state.points[i]) for i in range(24))
        player_flip = sum(max(0, flipped.points[i]) for i in range(24))
        opponent_orig = sum(abs(min(0, state.points[i])) for i in range(24))
        opponent_flip = sum(abs(min(0, flipped.points[i])) for i in range(24))
        assert player_flip == opponent_orig
        assert opponent_flip == player_orig

    def test_double_flip_identity(self):
        state = BoardState.standard_setup()
        double_flipped = flip_perspective(flip_perspective(state))
        np.testing.assert_array_equal(state.points, double_flipped.points)
        assert state.bar_player == double_flipped.bar_player
        assert state.bar_opponent == double_flipped.bar_opponent


# ── Dice and move generation tests ─────────────────────────────────

class TestDice:
    def test_roll_range(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            d1, d2 = roll_dice(rng)
            assert 1 <= d1 <= 6
            assert 1 <= d2 <= 6

    def test_doubles_give_four(self):
        assert len(get_dice_to_use(3, 3)) == 4

    def test_non_doubles_give_two(self):
        assert len(get_dice_to_use(3, 5)) == 2


class TestMoveGeneration:
    def test_opening_move_count(self):
        state = BoardState.standard_setup()
        plays = generate_plays(state, 3, 1)
        assert len(plays) > 0

    def test_no_legal_moves(self):
        state = BoardState()
        state.bar_player = 1
        for i in range(18, 24):
            state.points[i] = -2
        plays = generate_plays(state, 3, 1)
        assert len(plays) == 1
        assert plays[0].num_moves == 0

    def test_bearing_off(self):
        state = BoardState()
        state.points[5] = 5
        state.points[3] = 5
        state.points[1] = 5
        state.points[23] = -15
        plays = generate_plays(state, 6, 4)
        assert len(plays) > 0
        has_bear_off = any(
            any(m.dest == -1 for m in p.moves)
            for p in plays
        )
        assert has_bear_off


def _spread_race() -> BoardState:
    """BgMoveGen's pinned race (InteropTests.SpreadRace): one on-roll checker
    on each point from 2 to 16, the opponent's fifteen in its home board —
    three each on 19, 20 and 21, two each on 22, 23 and 24. Its 1-1 has 1,547
    distinct plays, far more than the wrapper's starting buffer holds."""
    state = BoardState()
    for point in range(2, 17):
        state.points[point - 1] = 1
    for point, count in ((19, 3), (20, 3), (21, 3), (22, 2), (23, 2), (24, 2)):
        state.points[point - 1] = -count
    return state


class TestSuccessorStates:
    """generate_successor_states against BgMoveGen's interop contract."""

    def test_every_successor_is_returned_past_the_starting_buffer(self):
        from engine.movegen import generate_successor_states
        successors = generate_successor_states(_spread_race(), 1, 1)
        assert len(successors) == 1547
        distinct = {
            (tuple(s.points), s.bar_player, s.bar_opponent,
             s.off_player, s.off_opponent)
            for s in successors
        }
        assert len(distinct) == 1547

    def test_a_malformed_board_raises_naming_invalid_position(self):
        from engine.movegen import generate_successor_states
        state = BoardState.standard_setup()
        state.bar_player = 1  # a sixteenth on-roll checker
        with pytest.raises(ValueError, match=r"INVALID_POSITION \(-2\)"):
            generate_successor_states(state, 3, 1)

    @pytest.mark.parametrize("die1, die2", [(0, 3), (7, 3), (3, 0), (3, 7)])
    def test_a_die_outside_one_to_six_raises_naming_invalid_argument(self, die1, die2):
        from engine.movegen import generate_successor_states
        with pytest.raises(ValueError, match=r"INVALID_ARGUMENT \(-1\)"):
            generate_successor_states(BoardState.standard_setup(), die1, die2)


# ── Network tests ──────────────────────────────────────────────────

class TestNetwork:
    def test_output_shape(self):
        import torch
        net = TDNetwork(hidden_layers=[64, 64])
        x = torch.randn(1, BOARD_FEATURE_SIZE)
        y = net(x)
        assert y.shape == (1, NUM_OUTPUTS)

    def test_output_range(self):
        import torch
        net = TDNetwork(hidden_layers=[64, 64])
        x = torch.randn(10, BOARD_FEATURE_SIZE)
        y = net(x)
        assert (y >= 0).all() and (y <= 1).all()

    def test_evaluate_convenience(self):
        import torch
        net = TDNetwork(hidden_layers=[64, 64])
        x = torch.randn(BOARD_FEATURE_SIZE)
        y = net.evaluate(x)
        assert y.shape == (NUM_OUTPUTS,)

    def test_from_state_dict_infers_architecture(self):
        import torch
        net = TDNetwork(hidden_layers=[64, 32])
        rebuilt = TDNetwork.from_state_dict(net.state_dict())
        assert rebuilt.input_size == BOARD_FEATURE_SIZE
        assert rebuilt.hidden_layers == [64, 32]
        x = torch.randn(4, BOARD_FEATURE_SIZE)
        assert torch.equal(net(x), rebuilt(x))

    def test_from_state_dict_rejects_wrong_output_size(self):
        # Only the first Linear (8 outputs) — a "final layer" of size 8 ≠ 6.
        sd = TDNetwork(input_size=10, hidden_layers=[8]).state_dict()
        bad = {k: v for k, v in sd.items() if k.startswith("network.0.")}
        with pytest.raises(ValueError):
            TDNetwork.from_state_dict(bad)

    def test_architecture_describes_the_network(self):
        net = TDNetwork(input_size=12, hidden_layers=[8, 4], dropout=0.5)
        # Dropout is deliberately absent: it leaves no trace in the weights.
        assert net.architecture == {
            "input_size": 12, "hidden_layers": [8, 4],
        }

    def test_architecture_is_a_copy(self):
        net = TDNetwork(hidden_layers=[64, 32])
        net.architecture["hidden_layers"].append(999)
        assert net.hidden_layers == [64, 32]

    def test_architecture_rebuilds_the_same_shape(self):
        net = TDNetwork(input_size=12, hidden_layers=[8, 4])
        assert TDNetwork(**net.architecture).architecture == net.architecture

    def test_from_state_dict_prefers_the_supplied_architecture(self):
        import torch
        net = TDNetwork(hidden_layers=[64, 32])
        rebuilt = TDNetwork.from_state_dict(
            net.state_dict(), architecture=net.architecture
        )
        assert rebuilt.architecture == net.architecture
        x = torch.randn(4, BOARD_FEATURE_SIZE)
        assert torch.equal(net(x), rebuilt(x))

    def test_from_state_dict_rejects_architecture_weight_mismatch(self):
        net = TDNetwork(hidden_layers=[64, 32])
        lying = {"input_size": BOARD_FEATURE_SIZE, "hidden_layers": [64, 16]}
        with pytest.raises(ValueError, match="disagrees"):
            TDNetwork.from_state_dict(net.state_dict(), architecture=lying)

    def test_from_state_dict_rejects_malformed_architecture(self):
        net = TDNetwork(hidden_layers=[64, 32])
        with pytest.raises(ValueError, match="malformed"):
            TDNetwork.from_state_dict(
                net.state_dict(), architecture={"hidden_layers": [64, 32]}
            )

    def test_equity_computation(self):
        import torch
        output = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        eq = compute_equity(output)
        assert eq.item() == pytest.approx(1.0)

        output = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        eq = compute_equity(output)
        assert eq.item() == pytest.approx(-2.0)


# ── Checkpoint contract ────────────────────────────────────────────

CHECKPOINT_HIDDEN_LAYERS = [16, 8]


def _read_raw(path):
    """The saved file's raw dict, for the contract tests to inspect by key.

    The raw `torch.load` here is deliberate and is not a reader: nothing
    is rebuilt from it. Every reader that rebuilds a network goes through
    `load_checkpoint`.
    """
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture
def saved_checkpoint(tmp_path):
    """A real checkpoint, written by the trainer's own save path.

    Returns the trainer's network, the saved file's path, and the file's
    raw dict.
    """
    import torch
    with open("configs/default.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["network"]["hidden_layers"] = CHECKPOINT_HIDDEN_LAYERS
    trainer = Trainer(config, torch.device("cpu"), tmp_path / "output")
    path = trainer._save_checkpoint("contract_test")
    return trainer.network, path, _read_raw(path)


def _load_edited(raw, tmp_path):
    """Write an edited checkpoint dict and load it the readers' one way."""
    import torch
    path = tmp_path / "edited.pt"
    torch.save(raw, path)
    return load_checkpoint(path)


class TestCheckpointArchitecture:
    """Both checkpoint generations must load, and must load correctly.

    A checkpoint written today embeds `TDNetwork.architecture`; one
    written before the contract existed carries no such key and is
    reconstructed by inferring the architecture from the weight shapes.
    """

    @staticmethod
    def _assert_matches(network, original):
        import torch
        assert network.architecture == original.architecture
        x = torch.randn(4, original.input_size)
        with torch.no_grad():
            assert torch.equal(network(x), original.cpu()(x))

    def test_save_embeds_the_architecture(self, saved_checkpoint):
        original, _, raw = saved_checkpoint
        assert raw[CHECKPOINT_ARCHITECTURE_KEY] == {
            "input_size": BOARD_FEATURE_SIZE,
            "hidden_layers": CHECKPOINT_HIDDEN_LAYERS,
        }
        assert raw[CHECKPOINT_ARCHITECTURE_KEY] == original.architecture

    def test_current_checkpoint_round_trips(self, saved_checkpoint):
        original, path, _ = saved_checkpoint
        self._assert_matches(load_checkpoint(path).network, original)

    def test_legacy_checkpoint_falls_back_to_inference(
        self, saved_checkpoint, tmp_path
    ):
        original, _, raw = saved_checkpoint
        legacy = {
            k: v for k, v in raw.items()
            if k != CHECKPOINT_ARCHITECTURE_KEY
        }
        assert CHECKPOINT_ARCHITECTURE_KEY not in legacy
        self._assert_matches(_load_edited(legacy, tmp_path).network, original)

    def test_corrupt_architecture_fails_loud(self, saved_checkpoint, tmp_path):
        _, _, raw = saved_checkpoint
        corrupt = dict(raw)
        corrupt[CHECKPOINT_ARCHITECTURE_KEY] = {
            "input_size": BOARD_FEATURE_SIZE,
            "hidden_layers": [size + 1 for size in CHECKPOINT_HIDDEN_LAYERS],
        }
        with pytest.raises(ValueError, match="disagrees"):
            _load_edited(corrupt, tmp_path)


class TestCheckpointEncodingVersion:
    """The encoding handshake: a checkpoint loads only under its encoding.

    A checkpoint written today stamps `ENCODING_VERSION`; one written
    before the stamp existed is read as encoding 1, with a warning.
    """

    def test_save_stamps_the_current_encoding_version(self, saved_checkpoint):
        from engine.state import ENCODING_VERSION
        _, _, raw = saved_checkpoint
        assert raw[CHECKPOINT_ENCODING_VERSION_KEY] == ENCODING_VERSION

    def test_save_stamps_the_live_constant(self, tmp_path, monkeypatch):
        # The stamp follows ENCODING_VERSION, not a literal that happens
        # to equal today's value.
        import torch
        monkeypatch.setattr(engine.checkpoint, "ENCODING_VERSION", 7)
        path = tmp_path / "stamped.pt"
        network = TDNetwork(hidden_layers=[4])
        engine.checkpoint.save_checkpoint(
            path, network, torch.optim.SGD(network.parameters(), lr=0.1),
            CheckpointStats(games_played=0, current_level=0, levels_reached=0),
        )
        assert _read_raw(path)[CHECKPOINT_ENCODING_VERSION_KEY] == 7

    def test_stamped_current_checkpoint_loads_silently(self, saved_checkpoint):
        import warnings
        _, path, _ = saved_checkpoint
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_checkpoint(path)

    def test_unstamped_checkpoint_loads_as_encoding_1_with_a_warning(
        self, saved_checkpoint, tmp_path
    ):
        original, _, raw = saved_checkpoint
        unstamped = {
            k: v for k, v in raw.items()
            if k != CHECKPOINT_ENCODING_VERSION_KEY
        }
        with pytest.warns(UserWarning, match=r"edited\.pt.*encoding version 1\b"):
            loaded = _load_edited(unstamped, tmp_path)
        assert loaded.network.architecture == original.architecture

    def test_unstamped_checkpoint_refuses_after_an_encoding_bump(
        self, saved_checkpoint, tmp_path, monkeypatch
    ):
        # Unstamped means encoding 1 — not "whatever is current".
        _, _, raw = saved_checkpoint
        unstamped = {
            k: v for k, v in raw.items()
            if k != CHECKPOINT_ENCODING_VERSION_KEY
        }
        monkeypatch.setattr(engine.checkpoint, "ENCODING_VERSION", 2)
        with pytest.warns(UserWarning, match="encoding version 1"):
            with pytest.raises(
                ValueError,
                match=r"encoding version 1 \(assumed.*encoding version 2\b",
            ):
                _load_edited(unstamped, tmp_path)

    def test_other_encoding_refuses_naming_both_versions(
        self, saved_checkpoint, tmp_path
    ):
        from engine.state import ENCODING_VERSION
        _, _, raw = saved_checkpoint
        other = ENCODING_VERSION + 1
        mismatched = dict(raw)
        mismatched[CHECKPOINT_ENCODING_VERSION_KEY] = other
        with pytest.raises(
            ValueError,
            match=rf"encoding version {other}\b.*"
                  rf"encoding version {ENCODING_VERSION}\b",
        ):
            _load_edited(mismatched, tmp_path)

    @pytest.mark.parametrize("stamp", ["1", 1.0, True, None])
    def test_malformed_encoding_stamp_fails_loud(
        self, saved_checkpoint, tmp_path, stamp
    ):
        _, _, raw = saved_checkpoint
        malformed = dict(raw)
        malformed[CHECKPOINT_ENCODING_VERSION_KEY] = stamp
        with pytest.raises(ValueError, match="malformed encoding version"):
            _load_edited(malformed, tmp_path)


class TestCheckpointStats:
    def test_stats_round_trip(self, saved_checkpoint):
        _, path, _ = saved_checkpoint
        # A freshly constructed trainer has made no progress.
        assert load_checkpoint(path).stats == CheckpointStats(
            games_played=0, current_level=0, levels_reached=0,
        )

    @pytest.mark.parametrize("edit", ["missing field", "non-integer"])
    def test_malformed_stats_fail_loud(self, saved_checkpoint, tmp_path, edit):
        _, _, raw = saved_checkpoint
        # The stats key is private to the format; spell it from its owner.
        stats_key = engine.checkpoint._STATS_KEY
        stats = dict(raw[stats_key])
        if edit == "missing field":
            del stats["games_played"]
        else:
            stats["games_played"] = "many"
        malformed = dict(raw)
        malformed[stats_key] = stats
        with pytest.raises(ValueError, match="malformed stats"):
            _load_edited(malformed, tmp_path)


# ── SPRT tests ─────────────────────────────────────────────────────

class TestSPRT:
    def test_early_strong_accept(self):
        result = sprt_test(wins=95, games=100)
        assert result == SPRTResult.ACCEPT

    def test_early_strong_reject(self):
        result = sprt_test(wins=30, games=100)
        assert result == SPRTResult.REJECT

    def test_continue_ambiguous(self):
        result = sprt_test(wins=73, games=100)
        assert result == SPRTResult.CONTINUE

    def test_hard_cap_rejects(self):
        result = sprt_test(wins=1460, games=2000)
        assert result == SPRTResult.REJECT

    def test_zero_games_continues(self):
        result = sprt_test(wins=0, games=0)
        assert result == SPRTResult.CONTINUE


# ── Result target encoding ─────────────────────────────────────────

class TestResultTarget:
    def test_win_target(self):
        from engine.game import GameResult
        target = result_to_target(GameResult.WIN)
        assert target[0] == 1.0
        assert sum(target) == 1.0

    def test_lose_gammon_target(self):
        from engine.game import GameResult
        target = result_to_target(GameResult.LOSE_GAMMON)
        assert target[4] == 1.0
        assert sum(target) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])