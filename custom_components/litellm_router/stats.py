"""Turn parsed Prometheus samples into per-backend statistics (pure logic)."""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any
from dataclasses import dataclass, field

from .api import Topology
from .const import IDLE
from .prometheus import Sample
from .tracker import CounterAccumulator

# A source (metric name + full label set) that has not been seen for this long and has
# no pending growth is rolled into the aggregates and dropped. Dropped sources leave a
# tombstone (their last raw reading) so a returning source continues without double
# counting; tombstones and live sources are both capped.
RETENTION_SECONDS = 24 * 3600
MAX_SOURCES = 5000
MAX_TOMBSTONES = 5000
STATE_VERSION = 1

REQUESTS = "litellm_proxy_total_requests_metric_total"
FAILURES = "litellm_proxy_failed_requests_metric_total"
LATENCY_SUM = "litellm_llm_api_latency_metric_sum"
LATENCY_COUNT = "litellm_llm_api_latency_metric_count"
TTFT_SUM = "litellm_llm_api_time_to_first_token_metric_sum"
TTFT_COUNT = "litellm_llm_api_time_to_first_token_metric_count"
TOKENS_IN = "litellm_input_tokens_metric_total"
TOKENS_OUT = "litellm_output_tokens_metric_total"

_ATTR_FIELDS = {
    REQUESTS: "requests",
    FAILURES: "failures",
    TOKENS_IN: "tokens_in",
    TOKENS_OUT: "tokens_out",
}
_FAMILIES = (
    REQUESTS,
    FAILURES,
    LATENCY_SUM,
    LATENCY_COUNT,
    TTFT_SUM,
    TTFT_COUNT,
    TOKENS_IN,
    TOKENS_OUT,
)


@dataclass(slots=True)
class BackendStats:
    """Statistics for one backend (model_id)."""

    requests: float = 0
    failures: float = 0
    tokens_in: float = 0
    tokens_out: float = 0
    avg_latency: float | None = None
    avg_ttft: float | None = None


@dataclass(slots=True)
class MetricsData:
    """Result of one metrics poll."""

    backends: dict[str, BackendStats] = field(default_factory=dict)
    fallback_requests: float = 0
    clients: dict[str, float] = field(default_factory=dict)
    active_backend: str = IDLE
    last_active_backend: str | None = None


def _served(labels: dict[str, str]) -> bool:
    status = labels.get("status_code")
    return status is None or status.startswith("2")


def backend_name(topology: Topology, model_id: str) -> str:
    """Return a unique, human-friendly name for a backend."""
    info = topology.models.get(model_id)
    if info is None:
        return model_id[:8]
    twins = [m for m in topology.models.values() if m.name == info.name]
    return info.name if len(twins) == 1 else f"{info.name} ({model_id[:6]})"


class MetricsProcessor:
    """Stateful processor: feed it one sample list per poll.

    Two stages for every counter family:

    1. ingest: update the per-source accumulator (metric name + all original labels)
       on every poll, whatever the topology, so resets are always seen;
    2. attribute: add the growth to the backend that resolves *now*. Growth that
       cannot be attributed (or, for requests, classified as fallback) yet stays
       pending per source and is applied exactly once later, even if the series vanishes.

    State can be exported and restored (see ``export_state``) so totals survive reloads.
    """

    def __init__(
        self, retention: float = RETENTION_SECONDS, max_sources: int = MAX_SOURCES
    ) -> None:
        """Initialise state."""
        self._retention = retention
        self._max_sources = max_sources
        self._reset()

    def _reset(self) -> None:
        self._acc = CounterAccumulator()
        self._latency: dict[str, float] = {}
        self._ttft: dict[str, float] = {}
        self._last_active: str | None = None
        self._fallback_total = 0.0
        self._attr: dict[tuple[str, str], float] = defaultdict(float)
        self._clients: dict[str, float] = defaultdict(float)
        self._pending_attr: dict[tuple, float] = defaultdict(float)
        self._pending_cls: dict[tuple, float] = defaultdict(float)
        self._last_seen: dict[tuple, float] = {}
        self._tombstones: dict[tuple, float] = {}

    @property
    def source_count(self) -> int:
        """Return the number of retained sources."""
        return len(self._last_seen)

    def _resolve(self, labels: dict[str, str], topology: Topology) -> str | None:
        """Return the backend model_id a source belongs to, if currently known."""
        model_id = labels.get("model_id", "")
        if model_id:
            return model_id
        # e.g. failure counters that only carry the requested model group
        group = labels.get("requested_model") or labels.get("model_group") or ""
        ids = [m.model_id for m in topology.models.values() if m.name == group]
        return ids[0] if len(ids) == 1 else None

    def process(
        self, samples: list[Sample], topology: Topology, now: float | None = None
    ) -> MetricsData:
        """Update state from samples and return the current data."""
        now = time.time() if now is None else now
        acc = self._acc
        poll_delta: dict[tuple, float] = {}

        # stage 1: ingest
        for s in samples:
            if s.name not in _FAMILIES:
                continue
            lb = s.labels
            if s.name == REQUESTS and (not _served(lb) or not lb.get("model_id")):
                continue
            key = (s.name, *sorted(lb.items()))
            if acc.last(key) is None and key in self._tombstones:
                # returning source: continue from where it was dropped (no double count)
                acc.restore(key, self._tombstones.pop(key), 0.0)
            before = acc.total(key)
            poll_delta[key] = acc.update(key, s.value)
            added = acc.total(key) - before
            self._last_seen[key] = now
            if s.name in _ATTR_FIELDS:
                self._pending_attr[key] += added
            if s.name == REQUESTS:
                self._pending_cls[key] += added
                self._clients[lb.get("client_ip", "None")] += added
        acc.finish_poll()

        # stage 2: attribute (all retained sources, including vanished ones)
        served_delta: dict[str, float] = defaultdict(float)
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, float] = defaultdict(float)
        t_sums: dict[str, float] = defaultdict(float)
        t_counts: dict[str, float] = defaultdict(float)
        targets = topology.fallback_targets
        for key in self._last_seen:
            name, lb = key[0], dict(key[1:])
            mid = self._resolve(lb, topology)
            if mid is None:
                continue
            self._attr[(name, mid)] += self._pending_attr.pop(key, 0.0)
            delta = poll_delta.get(key, 0.0)
            if name == REQUESTS:
                served_delta[mid] += delta
                info = topology.models.get(mid)
                if info is not None:
                    pending = self._pending_cls.pop(key, 0.0)
                    requested = lb.get("requested_model", "")
                    if (
                        requested
                        and requested != info.name
                        and (not targets or info.name in targets)
                    ):
                        self._fallback_total += pending
            elif name == LATENCY_SUM:
                sums[mid] += delta
            elif name == LATENCY_COUNT:
                counts[mid] += delta
            elif name == TTFT_SUM:
                t_sums[mid] += delta
            elif name == TTFT_COUNT:
                t_counts[mid] += delta

        for mid, n in counts.items():
            if n > 0:
                self._latency[mid] = sums[mid] / n
        for mid, n in t_counts.items():
            if n > 0:
                self._ttft[mid] = t_sums[mid] / n

        self._prune(now)
        return self._build(served_delta, topology)

    def _has_pending(self, key: tuple) -> bool:
        return bool(self._pending_attr.get(key) or self._pending_cls.get(key))

    def _drop(self, key: tuple) -> None:
        """Forget a source (its totals already live in the aggregates)."""
        self._tombstones[key] = self._acc.last(key)
        self._acc.forget(key)
        self._last_seen.pop(key, None)
        self._pending_attr.pop(key, None)
        self._pending_cls.pop(key, None)
        while len(self._tombstones) > MAX_TOMBSTONES:
            del self._tombstones[next(iter(self._tombstones))]

    def _prune(self, now: float) -> None:
        """Bound retained state; sources with pending growth are never dropped."""
        stale = [
            k
            for k, seen in self._last_seen.items()
            if now - seen > self._retention and not self._has_pending(k)
        ]
        for key in stale:
            self._drop(key)
        excess = len(self._last_seen) - self._max_sources
        if excess > 0:
            oldest = sorted(
                (k for k in self._last_seen if not self._has_pending(k)),
                key=lambda k: self._last_seen[k],
            )
            for key in oldest[:excess]:
                self._drop(key)

    def export_state(self) -> dict[str, Any]:
        """Return JSON-serialisable state."""
        acc = self._acc
        return {
            "version": STATE_VERSION,
            "primed": acc.primed,
            "sources": [
                {
                    "key": [key[0], *[list(p) for p in key[1:]]],
                    "last": acc.last(key),
                    "total": acc.total(key),
                    "seen": seen,
                    "pa": self._pending_attr.get(key, 0.0),
                    "pc": self._pending_cls.get(key, 0.0),
                }
                for key, seen in self._last_seen.items()
            ],
            "tombstones": [
                [[k[0], *[list(p) for p in k[1:]]], v] for k, v in self._tombstones.items()
            ],
            "attr": [[n, m, v] for (n, m), v in self._attr.items()],
            "clients": dict(self._clients),
            "fallback": self._fallback_total,
            "latency": self._latency,
            "ttft": self._ttft,
            "last_active": self._last_active,
        }

    def load_state(self, data: Any) -> bool:
        """Restore state; on any problem start fresh and return False."""
        try:
            if self._load_state(data):
                return True
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            pass
        self._reset()
        return False

    def _load_state(self, data: Any) -> bool:
        if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
            return False

        def to_key(raw: list) -> tuple:
            return (str(raw[0]), *[(str(k), str(v)) for k, v in raw[1:]])

        self._reset()
        for src in data["sources"]:
            key = to_key(src["key"])
            self._acc.restore(key, float(src["last"]), float(src["total"]))
            self._last_seen[key] = float(src["seen"])
            if src["pa"]:
                self._pending_attr[key] = float(src["pa"])
            if src["pc"]:
                self._pending_cls[key] = float(src["pc"])
        self._acc.set_primed(bool(data["primed"]))
        for raw, last in data["tombstones"]:
            self._tombstones[to_key(raw)] = float(last)
        for name, mid, value in data["attr"]:
            self._attr[(str(name), str(mid))] = float(value)
        self._clients.update({str(k): float(v) for k, v in data["clients"].items()})
        self._fallback_total = float(data["fallback"])
        self._latency = {str(k): float(v) for k, v in data["latency"].items()}
        self._ttft = {str(k): float(v) for k, v in data["ttft"].items()}
        last_active = data["last_active"]
        self._last_active = None if last_active is None else str(last_active)
        return True

    def _build(self, served_delta: dict[str, float], topology: Topology) -> MetricsData:
        data = MetricsData()
        totals: dict[str, BackendStats] = defaultdict(BackendStats)
        for (name, model_id), value in self._attr.items():
            if name in _ATTR_FIELDS:
                field_name = _ATTR_FIELDS[name]
                stats = totals[model_id]
                setattr(stats, field_name, getattr(stats, field_name) + value)
        for model_id, stats in totals.items():
            stats.avg_latency = self._latency.get(model_id)
            stats.avg_ttft = self._ttft.get(model_id)
        data.backends = dict(totals)
        data.clients = {c: v for c, v in self._clients.items() if c not in ("None", "")}
        data.fallback_requests = self._fallback_total
        if served_delta:
            best = max(served_delta, key=lambda m: served_delta[m])
            if served_delta[best] > 0:
                data.active_backend = backend_name(topology, best)
                self._last_active = data.active_backend
        data.last_active_backend = self._last_active
        return data
