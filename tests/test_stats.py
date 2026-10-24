"""Stats processing: deltas, resets, fallbacks, active backend."""

from custom_components.litellm_router.api import ModelInfo, Topology
from custom_components.litellm_router.prometheus import parse_prometheus
from custom_components.litellm_router.stats import MetricsProcessor, backend_name

from .conftest import FALLBACK, FAST, LOCAL, metric, metrics_text, req

TOPO = Topology(
    {
        LOCAL: ModelInfo(LOCAL, "qwen-local"),
        FALLBACK: ModelInfo(FALLBACK, "qwen-fallback"),
        FAST: ModelInfo(FAST, "qwen-fast"),
    },
    {"qwen-local": ["qwen-fallback"]},
)


def poll(proc, *lines, topo=TOPO):
    return proc.process(parse_prometheus(metrics_text(*lines)), topo)


def lat(mid, s, c):
    return [
        metric("litellm_llm_api_latency_metric_sum", mid, s),
        metric("litellm_llm_api_latency_metric_count", mid, c),
        metric("litellm_llm_api_latency_metric_bucket", mid, c, le="+Inf"),
    ]


def test_full_cycle() -> None:
    p = MetricsProcessor()
    d = poll(
        p,
        req(LOCAL, "qwen-local", 10),
        req(LOCAL, "qwen-local", 5, client="10.0.0.8"),
        req(LOCAL, "qwen-local", 1, client="None", status="200"),
        req(FALLBACK, "qwen-local", 4),  # fallback served
        req(FALLBACK, "qwen-fallback", 2),  # direct, not a fallback
        req(LOCAL, "qwen-local", 9, status="500"),  # not served
        req("", "", 99),  # health probe, ignored
        metric("litellm_input_tokens_metric_total", LOCAL, 100),
        metric("litellm_output_tokens_metric_total", LOCAL, 50),
        metric("litellm_proxy_failed_requests_metric_total", "", 3, requested_model="qwen-local"),
        metric("litellm_proxy_failed_requests_metric_total", "", 3, requested_model="nope"),
        metric("litellm_other_metric_total", LOCAL, 3),
        *lat(LOCAL, 20.0, 10),
        metric("litellm_llm_api_time_to_first_token_metric_sum", LOCAL, 5),
        metric("litellm_llm_api_time_to_first_token_metric_count", LOCAL, 10),
    )
    assert d.active_backend == "idle"  # baseline poll: no deltas
    assert d.last_active_backend is None
    assert d.backends[LOCAL].requests == 16
    assert d.backends[FALLBACK].requests == 6
    assert d.backends[LOCAL].tokens_in == 100 and d.backends[LOCAL].tokens_out == 50
    assert d.backends[LOCAL].failures == 3
    assert d.backends[LOCAL].avg_latency is None and d.backends[LOCAL].avg_ttft is None
    assert d.fallback_requests == 4
    assert d.clients == {"10.0.0.9": 16, "10.0.0.8": 5}

    # second poll: 4 more local (2s total latency), 1 more fallback, 2 fallback-requests
    d = poll(
        p,
        req(LOCAL, "qwen-local", 14),
        req(LOCAL, "qwen-local", 5, client="10.0.0.8"),
        req(FALLBACK, "qwen-local", 6),
        req(FALLBACK, "qwen-fallback", 2),
        *lat(LOCAL, 22.0, 14),
        metric("litellm_llm_api_time_to_first_token_metric_sum", LOCAL, 5.8),
        metric("litellm_llm_api_time_to_first_token_metric_count", LOCAL, 14),
    )
    assert d.active_backend == "qwen-local"  # 4 vs 2
    assert d.backends[LOCAL].requests == 20
    assert d.fallback_requests == 6
    assert d.backends[LOCAL].avg_latency == 0.5
    assert abs(d.backends[LOCAL].avg_ttft - 0.2) < 1e-9

    # idle poll: latency carried forward
    d = poll(p, req(LOCAL, "qwen-local", 14), *lat(LOCAL, 22.0, 14))
    assert d.active_backend == "idle"
    assert d.last_active_backend == "qwen-local"
    assert d.backends[LOCAL].avg_latency == 0.5
    assert d.backends[LOCAL].requests == 20  # backend/series absent stays put


def test_counter_reset_never_negative_and_backend_return() -> None:
    p = MetricsProcessor()
    poll(p, req(LOCAL, "qwen-local", 100), *lat(LOCAL, 50.0, 100))
    d = poll(p, req(LOCAL, "qwen-local", 3), *lat(LOCAL, 6.0, 3))  # container restarted
    assert d.backends[LOCAL].requests == 103
    assert d.active_backend == "qwen-local"
    assert d.backends[LOCAL].avg_latency == 2.0
    # backend vanishes, then returns higher than before
    poll(p)
    d = poll(p, req(LOCAL, "qwen-local", 8))
    assert d.backends[LOCAL].requests == 108


def test_fallback_without_known_targets_and_unknown_backend() -> None:
    p = MetricsProcessor()
    topo = Topology({LOCAL: ModelInfo(LOCAL, "qwen-local")})
    d = poll(p, req(LOCAL, "other", 2), req("zzzzzzzzzz", "qwen-local", 1), topo=topo)
    assert d.fallback_requests == 2  # no fallback config: any requested != served counts
    p2 = MetricsProcessor()
    poll(p2, req("zzzzzzzzzz", "x", 1), topo=topo)
    d = poll(p2, req("zzzzzzzzzz", "x", 2), topo=topo)
    assert d.active_backend == "zzzzzzzz"


def test_backend_name_disambiguates_twins() -> None:
    topo = Topology({LOCAL: ModelInfo(LOCAL, "same"), FAST: ModelInfo(FAST, "same")})
    assert backend_name(topo, LOCAL) == "same (aaaaaa)"
    assert backend_name(topo, "q" * 20) == "qqqqqqqq"


def test_ambiguous_group_resolution_skipped() -> None:
    topo = Topology({LOCAL: ModelInfo(LOCAL, "g"), FAST: ModelInfo(FAST, "g")})
    p = MetricsProcessor()
    d = poll(
        p,
        metric("litellm_proxy_failed_requests_metric_total", "", 1, requested_model="g"),
        topo=topo,
    )
    assert d.backends == {}
