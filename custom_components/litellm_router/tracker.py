"""Reset-safe counter accumulation."""

from __future__ import annotations

from collections.abc import Hashable


class CounterAccumulator:
    """Turn raw, resettable counters into monotonic totals and per-poll deltas.

    * The first poll only sets a baseline (the total starts at the raw value, delta 0).
    * A series first seen after that started at zero, so its whole value is the delta.
    * A value lower than the last one means the source restarted: the new value is
      the delta (never negative).
    * Series missing from a poll keep their state, so a returning backend continues
      from where it left off (and is handled as a reset if its counter went down).
    """

    def __init__(self) -> None:
        """Initialise empty state."""
        self._last: dict[Hashable, float] = {}
        self._total: dict[Hashable, float] = {}
        self._primed = False

    @property
    def primed(self) -> bool:
        """Return True once the baseline poll has been taken."""
        return self._primed

    def update(self, key: Hashable, raw: float) -> float:
        """Record a raw reading and return the non-negative delta."""
        last = self._last.get(key)
        if last is None:
            delta = raw if self._primed else 0.0
            self._total[key] = raw
        else:
            delta = raw - last if raw >= last else raw
            self._total[key] += delta
        self._last[key] = raw
        return delta

    def finish_poll(self) -> None:
        """Mark the end of a poll; later new series count in full."""
        self._primed = True

    def total(self, key: Hashable) -> float:
        """Return the accumulated total for a series."""
        return self._total.get(key, 0.0)

    def series(self) -> list[Hashable]:
        """Return every series seen so far."""
        return list(self._total)

    def last(self, key: Hashable) -> float | None:
        """Return the last raw reading, if any."""
        return self._last.get(key)

    def restore(self, key: Hashable, last: float, total: float) -> None:
        """Restore a series from persisted state."""
        self._last[key] = last
        self._total[key] = total

    def forget(self, key: Hashable) -> None:
        """Drop a series."""
        self._last.pop(key, None)
        self._total.pop(key, None)

    def set_primed(self, primed: bool) -> None:
        """Restore the primed flag."""
        self._primed = primed
