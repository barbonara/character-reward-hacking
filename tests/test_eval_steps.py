"""Tests for get_eval_steps function."""

import pytest
from src.tinker_local.train import get_eval_steps


class TestGetEvalStepsNumEvals:
    """Tests for num_evals parameter."""

    def test_num_evals_zero(self):
        assert get_eval_steps(100, num_evals=0) == set()

    def test_num_evals_one_final_only(self):
        assert get_eval_steps(100, num_evals=1) == {99}

    def test_num_evals_two_first_and_final(self):
        assert get_eval_steps(100, num_evals=2) == {0, 99}

    def test_num_evals_three_first_middle_final(self):
        # 0, 49.5 -> 50, 99
        assert get_eval_steps(100, num_evals=3) == {0, 50, 99}

    def test_num_evals_four(self):
        # 0, 33, 66, 99
        assert get_eval_steps(100, num_evals=4) == {0, 33, 66, 99}

    def test_num_evals_five(self):
        # 0, 24.75 -> 25, 49.5 -> 50, 74.25 -> 74, 99
        assert get_eval_steps(100, num_evals=5) == {0, 25, 50, 74, 99}

    def test_num_evals_small_total_steps(self):
        # 10 steps, 3 evals: 0, 4.5 -> 4, 9
        assert get_eval_steps(10, num_evals=3) == {0, 4, 9}

    def test_num_evals_equals_total_steps(self):
        # Every step is an eval
        assert get_eval_steps(5, num_evals=5) == {0, 1, 2, 3, 4}

    def test_num_evals_more_than_total_steps(self):
        # num_evals > total_steps still works, some steps may overlap
        result = get_eval_steps(5, num_evals=10)
        assert 0 in result
        assert 4 in result


class TestGetEvalStepsEvalEvery:
    """Tests for eval_every parameter."""

    def test_eval_every_zero_disabled(self):
        assert get_eval_steps(100, eval_every=0) == set()

    def test_eval_every_ten(self):
        # 0, 10, 20, ..., 90, 99 (final added)
        expected = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99}
        assert get_eval_steps(100, eval_every=10) == expected

    def test_eval_every_includes_final(self):
        # Even if eval_every doesn't land on final, it's included
        result = get_eval_steps(100, eval_every=30)
        assert 99 in result
        assert result == {0, 30, 60, 90, 99}

    def test_eval_every_larger_than_total(self):
        # Only step 0 and final
        assert get_eval_steps(50, eval_every=100) == {0, 49}

    def test_eval_every_one(self):
        # Every step
        assert get_eval_steps(5, eval_every=1) == {0, 1, 2, 3, 4}


class TestGetEvalStepsEdgeCases:
    """Edge cases and parameter interaction."""

    def test_no_params_returns_empty(self):
        assert get_eval_steps(100) == set()

    def test_num_evals_takes_precedence(self):
        # When both are provided, num_evals is used
        result = get_eval_steps(100, num_evals=2, eval_every=10)
        assert result == {0, 99}

    def test_single_step_total(self):
        assert get_eval_steps(1, num_evals=1) == {0}
        assert get_eval_steps(1, num_evals=2) == {0}
        assert get_eval_steps(1, eval_every=1) == {0}

    def test_negative_num_evals(self):
        assert get_eval_steps(100, num_evals=-1) == set()
