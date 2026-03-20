"""Tests for BgRLEngine core modules."""

import numpy as np
import pytest

from engine.state import (
    BoardState, encode_board, encode_point, encode_bar,
    encode_borne_off, flip_perspective,
    BOARD_FEATURE_SIZE, NUM_POINTS, UNITS_PER_POINT,
)
from engine.setup_generator import SetupGenerator, QUADRANTS
from engine.dice import roll_dice, generate_plays, get_dice_to_use
from engine.network import TDNetwork, compute_equity, NUM_OUTPUTS
from training.td_trainer import sprt_test, SPRTResult, result_to_target


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
        state = BoardState.nackgammon_setup()
        player = sum(max(0, state.points[i]) for i in range(24))
        opponent = sum(abs(min(0, state.points[i])) for i in range(24))
        assert player == 15
        assert opponent == 15

    def test_standard_pip_count(self):
        state = BoardState.standard_setup()
        # Standard bg pip count = 167
        assert state.player_pip_count() == 167
        assert state.opponent_pip_count() == 167

    def test_is_race_standard(self):
        state = BoardState.standard_setup()
        assert not state.is_race()

    def test_is_race_separated(self):
        state = BoardState()
        # All player checkers in home board, all opponent in their home board
        state.points[0] = 5   # player on point 1
        state.points[3] = 5   # player on point 4
        state.points[5] = 5   # player on point 6
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
        # After flip, what was opponent's checkers become player's
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


# ── Setup generator tests ──────────────────────────────────────────

class TestSetupGenerator:
    def setup_method(self):
        self.gen = SetupGenerator(rng=np.random.default_rng(42))

    def test_generates_valid_position(self):
        state = self.gen.generate()
        player = sum(max(0, state.points[i]) for i in range(24))
        assert player == 15

    def test_symmetry(self):
        state = self.gen.generate()
        for i in range(24):
            mirror = 23 - i
            player_here = max(0, state.points[i])
            opp_mirror = abs(min(0, state.points[mirror]))
            assert player_here == opp_mirror, (
                f"Asymmetry at point {i}: player={player_here}, "
                f"opponent at mirror {mirror}={opp_mirror}"
            )

    def test_min_checkers_per_point(self):
        for _ in range(50):
            state = self.gen.generate()
            for i in range(24):
                if state.points[i] > 0:
                    assert state.points[i] >= 2

    def test_quadrant_coverage(self):
        for _ in range(50):
            state = self.gen.generate()
            for q in QUADRANTS:
                has_checker = any(state.points[i] > 0 for i in q)
                assert has_checker, f"Quadrant {list(q)} has no player checkers"

    def test_min_pip_count(self):
        for _ in range(50):
            state = self.gen.generate()
            pips = state.player_pip_count()
            assert pips >= 100

    def test_standard_is_valid(self):
        state = self.gen.standard()
        assert sum(max(0, state.points[i]) for i in range(24)) == 15

    def test_nackgammon_is_valid(self):
        state = self.gen.nackgammon()
        assert sum(max(0, state.points[i]) for i in range(24)) == 15

    def test_batch_generation(self):
        batch = self.gen.generate_batch(10)
        assert len(batch) == 10

    def test_weight_update(self):
        self.gen.set_weights({4: 1, 5: 1, 6: 1, 7: 1})
        state = self.gen.generate()
        player = sum(max(0, state.points[i]) for i in range(24))
        assert player == 15


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
        # 3-1 opening: known to have specific number of legal plays
        plays = generate_plays(state, 3, 1)
        # Should have at least a few plays
        assert len(plays) > 0

    def test_no_legal_moves(self):
        # Create a position where the player is on the bar
        # and opponent holds the entire entry area
        state = BoardState()
        state.bar_player = 1
        # Opponent makes all 6 points in their home board (player's 19-24)
        for i in range(18, 24):
            state.points[i] = -2
        plays = generate_plays(state, 3, 1)
        # Should get an empty play (no moves possible)
        assert len(plays) == 1
        assert plays[0].num_moves == 0

    def test_bearing_off(self):
        state = BoardState()
        state.points[5] = 5  # 5 on the 6-point
        state.points[3] = 5  # 5 on the 4-point
        state.points[1] = 5  # 5 on the 2-point
        # Put opponent checkers far away (no contact)
        state.points[23] = -15
        plays = generate_plays(state, 6, 4)
        assert len(plays) > 0
        # At least one play should bear off a checker
        has_bear_off = any(
            any(m.dest == -1 for m in p.moves)
            for p in plays
        )
        assert has_bear_off


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

    def test_equity_computation(self):
        import torch
        # Pure win: P(win)=1, rest=0
        output = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        eq = compute_equity(output)
        assert eq.item() == pytest.approx(1.0)

        # Pure gammon loss
        output = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        eq = compute_equity(output)
        assert eq.item() == pytest.approx(-2.0)


# ── SPRT tests ─────────────────────────────────────────────────────

class TestSPRT:
    def test_early_strong_accept(self):
        # Overwhelming wins should accept quickly
        result = sprt_test(wins=95, games=100)
        assert result == SPRTResult.ACCEPT

    def test_early_strong_reject(self):
        # Overwhelming losses should reject quickly
        result = sprt_test(wins=30, games=100)
        assert result == SPRTResult.REJECT

    def test_continue_ambiguous(self):
        # Close to boundary should continue
        result = sprt_test(wins=73, games=100)
        assert result == SPRTResult.CONTINUE

    def test_hard_cap_rejects(self):
        # At hard cap, should reject
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
