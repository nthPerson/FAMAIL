# famail_temporal/tests/test_config_multiloop.py
"""Defaults for the multi-loop re-attribution + acceptance-gate knobs."""
from famail_temporal import config


def test_multiloop_defaults_are_backward_compatible():
    # max_rounds=1 ⇒ today's single pass; objective gate ⇒ today's acceptance;
    # epsilon_cap == EPSILON_BALL ⇒ no extra clip for single edits.
    assert config.MAX_ROUNDS == 1
    assert config.ROUND_CONVERGENCE_TOL is None
    assert config.ROUND_PATIENCE == 2
    assert config.EPSILON_CAP == config.EPSILON_BALL
    assert config.ACCEPT_RULE == "objective"
    assert config.ITERATIVE_TOPK_MAX_EDITS == 1
