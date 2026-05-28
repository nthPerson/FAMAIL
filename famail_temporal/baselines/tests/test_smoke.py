"""Smoke test: the baselines package imports."""


def test_baselines_package_imports():
    import famail_temporal.baselines as b
    assert b is not None
