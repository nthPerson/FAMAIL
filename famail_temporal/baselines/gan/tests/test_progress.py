"""Unit tests for gan.progress (the non-TTY fallback + no-op behavior)."""
from famail_temporal.baselines.gan import progress as pg


def test_disabled_progress_is_silent(capsys):
    bar = pg.Progress(10, "phase", enabled=False)
    for _ in range(10):
        bar.update(1, loss="1.0")
    bar.close()
    assert capsys.readouterr().out == ""


def test_fallback_prints_periodically_when_not_tty(capsys, monkeypatch):
    # Force the non-TTY path so we exercise the print fallback, not tqdm.
    monkeypatch.setattr(pg, "_bars_enabled", lambda: False)
    bar = pg.Progress(10, "MLE epoch 1/1", enabled=True, print_every_frac=0.5)
    for _ in range(10):
        bar.update(1, loss="2.345")
    bar.close()
    out = capsys.readouterr().out
    # print_every_frac=0.5 -> a line every 5 updates (at 5 and 10).
    lines = [ln for ln in out.splitlines() if ln]
    assert len(lines) == 2
    assert "MLE epoch 1/1" in lines[0]
    assert "10/10 (100%)" in lines[-1]
    assert "loss=2.345" in lines[-1]


def test_elapsed_format():
    # 0 seconds elapsed formats as HH:MM:SS.
    import time
    assert pg.elapsed(time.monotonic()) == "00:00:00"
