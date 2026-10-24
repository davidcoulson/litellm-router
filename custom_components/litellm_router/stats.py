"""Turn parsed Prometheus samples into per-backend statistics (pure logic)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .api import Topology
from .const import IDLE
from .prometheus import Sample
from .tracker import CounterAccumulator

REQUESTS = "litellm_proxy_total_requests_metric_total"
FAILURES = "litellm_proxy_failed_requests_metric_total"
LATENCY_SUM = "litellm_llm_api_latency_metric_sum"
LATENCY_COUNT = "litellm_llm_api_latency_metric_count"
TTFT_SUM = "litellm_llm_api_time_to_first_token_metric_sum"
TTFT_COUNT = "litellm_llm_api_time_to_first_token_metric_count"
TOKENS_IN = "litellm_input_tokens_metric_total"
TOKENS_OUT = "litellm_output_tokens_metric_total"

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
    """Stateful processor: feed it one sample list per poll."""

    def __init__(self) -> None:
        """Initialise state."""
        self._acc = CounterAccumulator()
        self._latency: dict[str, float] = {}
        self._ttft: dict[str, float] = {}
        self._last_active: str | None = None

    def _resolve(self, labels: dict[str, str], topology: Topology) -> str | None:
        """Return the backend model_id a sample belongs to, if any."""
        model_id = labels.get("model_id", "")
        if model_id:
            return model_id
        # e.g. failure counters that only carry the requested model group
        group = labels.get("requested_model") or labels.get("model_group") or ""
        ids = [m.model_id for m in topology.models.values() if m.name == group]
        return ids[0] if len(ids) == 1 else None

    def process(self, samples: list[Sample], topology: Topology) -> MetricsData:
        """Update state from samples and return the current data."""
        acc = self._acc
        served_delta: dict[str, float] = defaultdict(float)
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, float] = defaultdict(float)
        t_sums: dict[str, float] = defaultdict(float)
        t_counts: dict[str, float] = defaultdict(float)

        for s in samples:
            if s.name not in _FAMILIES:
                continue
            model_id = self._resolve(s.labels, topology)
            if model_id is None:
                continue
            lb = s.labels
            if s.name == REQUESTS:
                if not _served(lb) or not lb.get("model_id"):
                    continue
                client = lb.get("client_ip", "None")
                key = (
                    s.name,
                    model_id,
                    lb.get("requested_model", ""),
                    client,
                    lb.get("status_code", ""),
                    lb.get("route", ""),
                    lb.get("user_agent", ""),
                )
                delta = acc.update(key, s.value)
                served_delta[model_id] += delta
                continue
            key = (s.name, model_id, *sorted(lb.items()))
            delta = acc.update(key, s.value)
            if s.name == LATENCY_SUM:
                sums[model_id] += delta
            elif s.name == LATENCY_COUNT:
                counts[model_id] += delta
            elif s.name == TTFT_SUM:
                t_sums[model_id] += delta
            elif s.name == TTFT_COUNT:
                t_counts[model_id] += delta
        acc.finish_poll()

        for mid, n in counts.items():
            if n > 0:
                self._latency[mid] = sums[mid] / n
        for mid, n in t_counts.items():
            if n > 0:
                self._ttft[mid] = t_sums[mid] / n

        return self._build(served_delta, topology)

    def _build(self, served_delta: dict[str, float], topology: Topology) -> MetricsData:
        acc = self._acc
        data = MetricsData()
        totals: dict[str, BackendStats] = defaultdict(BackendStats)
        clients: dict[str, float] = defaultdict(float)
        targets = topology.fallback_targets
        fallbacks = 0.0
        for key in acc.series():
            name, model_id = key[0], key[1]
            stats = totals[model_id]
            value = acc.total(key)
            if name == REQUESTS:
                stats.requests += value
                clients[key[3]] += value
                info = topology.models.get(model_id)
                group, requested = (info.name if info else ""), key[2]
                if requested and group and requested != group and (not targets or group in targets):
                    fallbacks += value
            elif name == FAILURES:
                stats.failures += value
            elif name == TOKENS_IN:
                stats.tokens_in += value
            elif name == TOKENS_OUT:
                stats.tokens_out += value
        for model_id, stats in totals.items():
            stats.avg_latency = self._latency.get(model_id)
            stats.avg_ttft = self._ttft.get(model_id)
        data.backends = dict(totals)
        data.clients = {c: v for c, v in clients.items() if c not in ("None", "")}
        data.fallback_requests = fallbacks
        if served_delta:
            best = max(served_delta, key=lambda m: served_delta[m])
            if served_delta[best] > 0:
                data.active_backend = backend_name(topology, best)
                self._last_active = data.active_backend
        data.last_active_backend = self._last_active
        return data
