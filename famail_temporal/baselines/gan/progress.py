"""Optional progress reporting for the GAN baseline runs.

Renders a tqdm bar on an interactive terminal, falls back to periodic
timestamped prints when stderr is not a TTY (e.g. output redirected to a log),
and is a no-op when disabled (so library/test calls stay silent). Mirrors the
convention in famail_temporal/evaluation/runner.py.
"""
from __future__ import annotations
import sys
import time

try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - tqdm is a normal dependency
    _TQDM_AVAILABLE = False
    _tqdm = None  # type: ignore


def _bars_enabled() -> bool:
    """tqdm bars only on a real terminal; otherwise fall back to prints."""
    return _TQDM_AVAILABLE and sys.stderr.isatty()


def elapsed(t0: float) -> str:
    """Format seconds-since-t0 as HH:MM:SS."""
    dt = time.monotonic() - t0
    h, rem = divmod(int(dt), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def log_phase(t0: float, msg: str) -> None:
    """One-line elapsed-time phase marker: ``[b0-adv +HH:MM:SS] msg``."""
    print(f"[b0-adv +{elapsed(t0)}] {msg}", flush=True)


class Progress:
    """Count up to ``total``, rendering a tqdm bar (interactive) or periodic
    timestamped prints (non-TTY). A no-op when ``enabled`` is False.

    Usage:
        with Progress(n_batches, "MLE epoch 1/5", enabled=progress) as p:
            for ...:
                p.update(1, loss=f"{loss:.3f}")
    """

    def __init__(
        self, total: int, desc: str, *, enabled: bool, print_every_frac: float = 0.1,
    ):
        self.enabled = enabled
        self.total = max(1, int(total))
        self.desc = desc
        self.n = 0
        self.t0 = time.monotonic()
        self._every = max(1, int(self.total * print_every_frac))
        self._bar = (
            _tqdm(total=self.total, desc=desc, file=sys.stderr,
                  dynamic_ncols=True, leave=True)
            if (enabled and _bars_enabled()) else None
        )

    def update(self, n: int = 1, **postfix) -> None:
        if not self.enabled:
            return
        self.n += n
        if self._bar is not None:
            if postfix:
                self._bar.set_postfix(postfix, refresh=False)
            self._bar.update(n)
        elif self.n % self._every == 0 or self.n >= self.total:
            pct = 100.0 * self.n / self.total
            pf = " ".join(f"{k}={v}" for k, v in postfix.items())
            print(
                f"[{self.desc} +{elapsed(self.t0)}] {self.n}/{self.total} "
                f"({pct:.0f}%) {pf}".rstrip(),
                flush=True,
            )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
