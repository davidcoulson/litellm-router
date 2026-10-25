"""Stats processing: deltas, resets, fallbacks, active backend."""

import pytest

from custom_components.litellm_router.api import ModelInfo, Topology
from custom_components.litellm_router.prometheus import parse_prometheus
from custom_components.litellm_router.stats import BackendStats, MetricsProcessor, backend_name

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


def test_extra_labels_do_not_collide() -> None:
    p = MetricsProcessor()
    a = req(LOCAL, "qwen-local", 10).replace("client_ip=", 'hashed_api_key="k1",client_ip=')
    b = req(LOCAL, "qwen-local", 20).replace("client_ip=", 'hashed_api_key="k2",client_ip=')
    for _ in range(3):
        d = poll(p, a, b)
        assert d.backends[LOCAL].requests == 30
        assert d.clients == {"10.0.0.9": 30}
    assert d.active_backend == "idle"


def test_fallback_history_survives_topology_changes() -> None:
    p = MetricsProcessor()
    line = lambda v: req(FALLBACK, "qwen-local", v)  # noqa: E731
    assert poll(p, line(10)).fallback_requests == 10
    # backend removed from topology: history is kept, new traffic stays pending
    assert poll(p, line(12), topo=Topology()).fallback_requests == 10
    # restored: the pending amount is classified once, plus the new delta
    assert poll(p, line(15)).fallback_requests == 15
    # fallback config lost: unknown targets -> any requested != served counts, never decreases
    no_cfg = Topology(TOPO.models)
    assert poll(p, line(16), topo=no_cfg).fallback_requests == 16


def test_degraded_startup_recovers_fallbacks_once() -> None:
    p = MetricsProcessor()
    line = lambda v: req(FALLBACK, "qwen-local", v)  # noqa: E731
    assert poll(p, line(12), topo=Topology()).fallback_requests == 0
    assert poll(p, line(12), topo=Topology()).fallback_requests == 0
    assert poll(p, line(12)).fallback_requests == 12  # recovered
    assert poll(p, line(12)).fallback_requests == 12  # not counted twice
    assert poll(p, line(13)).fallback_requests == 13


def test_pending_series_recovered_after_series_disappears() -> None:
    p = MetricsProcessor()
    assert poll(p, req(FALLBACK, "qwen-local", 12), topo=Topology()).fallback_requests == 0
    # proxy restarted: the old series (different label) is gone
    other = req(FALLBACK, "qwen-local", 2).replace("curl", "curl2")
    assert poll(p, other, topo=Topology()).fallback_requests == 0
    assert poll(p).fallback_requests == 14  # topology back, series absent
    assert poll(p).fallback_requests == 14  # exactly once
    assert poll(p, other).fallback_requests == 14


def test_failure_attribution_does_not_double_count_on_deployment_swap() -> None:
    p = MetricsProcessor()
    fail = lambda v: metric(
        "litellm_proxy_failed_requests_metric_total", "", v, requested_model="g"
    )  # noqa: E731
    topo_a = Topology({LOCAL: ModelInfo(LOCAL, "g")})
    topo_b = Topology({FAST: ModelInfo(FAST, "g")})
    d = poll(p, fail(10), topo=topo_a)
    assert d.backends[LOCAL].failures == 10
    d = poll(p, fail(10), topo=topo_b)  # counter unchanged
    assert d.backends[LOCAL].failures == 10
    assert d.backends.get(FAST, BackendStats()).failures == 0
    d = poll(p, fail(12), topo=topo_b)  # only the growth goes to B
    assert d.backends[LOCAL].failures == 10 and d.backends[FAST].failures == 2


def _fail(v):
    return metric("litellm_proxy_failed_requests_metric_total", "", v, requested_model="g")


def test_reset_during_unresolved_poll_is_counted() -> None:
    p = MetricsProcessor()
    topo_a = Topology({LOCAL: ModelInfo(LOCAL, "g")})
    topo_b = Topology({FAST: ModelInfo(FAST, "g")})
    assert poll(p, _fail(10), topo=topo_a).backends[LOCAL].failures == 10
    d = poll(p, _fail(3), topo=Topology())  # reset seen while unresolved
    assert d.backends[LOCAL].failures == 10
    d = poll(p, _fail(12), topo=topo_b)  # 3 + 9 pending, total source 22
    assert d.backends[LOCAL].failures == 10 and d.backends[FAST].failures == 12
    d = poll(p, _fail(12), topo=topo_b)  # unchanged
    assert d.backends[FAST].failures == 12


def test_pending_applies_once_when_series_vanishes() -> None:
    p = MetricsProcessor()
    topo = Topology({LOCAL: ModelInfo(LOCAL, "g")})
    poll(p, _fail(1), topo=Topology())
    poll(p, _fail(5), topo=Topology())
    poll(p, topo=Topology())
    d = poll(p, topo=topo)  # series gone, topology back
    assert d.backends[LOCAL].failures == 5
    assert poll(p, topo=topo).backends[LOCAL].failures == 5


def _json_roundtrip(p):
    import json

    return json.loads(json.dumps(p.export_state()))


def test_state_roundtrip_keeps_totals_and_pending() -> None:
    p = MetricsProcessor()
    poll(p, req(LOCAL, "qwen-local", 100), req(FALLBACK, "qwen-local", 5), topo=Topology())
    poll(p, req(LOCAL, "qwen-local", 3), req(FALLBACK, "qwen-local", 5), topo=Topology())
    p.process(parse_prometheus(metrics_text(*lat(LOCAL, 2.0, 4))), TOPO)
    p2 = MetricsProcessor()
    assert p2.load_state(_json_roundtrip(p))
    # unchanged scrape after reload: nothing new, totals held, pending classified once
    d = poll(p2, req(LOCAL, "qwen-local", 3), req(FALLBACK, "qwen-local", 5))
    assert d.backends[LOCAL].requests == 103
    assert d.fallback_requests == 5
    assert d.last_active_backend == "aaaaaaaa"
    d = poll(p2, req(LOCAL, "qwen-local", 3), req(FALLBACK, "qwen-local", 5))
    assert d.fallback_requests == 5 and d.clients == {"10.0.0.9": 108}
    d = poll(p2, req(LOCAL, "qwen-local", 4), req(FALLBACK, "qwen-local", 5))
    assert d.backends[LOCAL].requests == 104 and d.active_backend == "qwen-local"
    # last_active and averages are restored too
    p3 = MetricsProcessor()
    assert p3.load_state(_json_roundtrip(p2))
    assert poll(p3, req(LOCAL, "qwen-local", 4)).last_active_backend == "qwen-local"


@pytest.mark.parametrize(
    "bad",
    [None, [], {"version": 99}, {"version": 1}, {"version": 1, "sources": [{"key": []}]}],
)
def test_load_state_tolerates_bad_data(bad) -> None:
    p = MetricsProcessor()
    poll(p, req(LOCAL, "qwen-local", 1))
    assert not p.load_state(bad)
    assert p.source_count == 0
    assert poll(p, req(LOCAL, "qwen-local", 2)).backends[LOCAL].requests == 2


def test_retention_bounds_state_under_churn() -> None:
    p = MetricsProcessor(retention=100, max_sources=10)
    now = 0.0
    for i in range(300):
        now += 60
        line = req(LOCAL, "qwen-local", 1, client=f"10.1.{i // 250}.{i % 250}")
        p.process(parse_prometheus(metrics_text(line)), TOPO, now=now)
        assert p.source_count <= 10
    d = p.process([], TOPO, now=now + 1)
    assert d.backends[LOCAL].requests == 300  # nothing lost when sources were dropped
    assert len(d.clients) == 300
    assert len(p.export_state()["sources"]) <= 10


def test_returning_dropped_source_is_not_double_counted() -> None:
    p = MetricsProcessor(retention=100)
    line = lambda v: req(LOCAL, "qwen-local", v)  # noqa: E731
    p.process(parse_prometheus(metrics_text(line(10))), TOPO, now=0)
    p.process([], TOPO, now=1000)
    assert p.source_count == 0
    d = p.process(parse_prometheus(metrics_text(line(12))), TOPO, now=1001)
    assert d.backends[LOCAL].requests == 12  # 10 + growth of 2
    # restart while dropped: a lower value is a reset
    p.process([], TOPO, now=5000)
    d = p.process(parse_prometheus(metrics_text(line(3))), TOPO, now=5001)
    assert d.backends[LOCAL].requests == 15


def test_pending_sources_are_never_dropped() -> None:
    p = MetricsProcessor(retention=100, max_sources=10)
    for i in range(3):
        p.process(
            parse_prometheus(metrics_text(req(FALLBACK, "qwen-local", 2, client=f"c{i}"))),
            Topology(),
            now=i,
        )
    p.process([], Topology(), now=10_000)
    assert p.source_count == 3
    d = p.process([], TOPO, now=10_001)  # topology back: classified, then droppable
    assert d.fallback_requests == 6
    p.process([], TOPO, now=20_000)
    p.process([], TOPO, now=20_001)
    assert p.source_count == 0


def test_tombstones_are_capped(monkeypatch) -> None:
    from custom_components.litellm_router import stats

    monkeypatch.setattr(stats, "MAX_TOMBSTONES", 2)
    p = MetricsProcessor(retention=1)
    for i in range(4):
        p.process(
            parse_prometheus(metrics_text(req(LOCAL, "qwen-local", 1, client=f"c{i}"))),
            TOPO,
            now=i * 10,
        )
    p.process([], TOPO, now=1000)
    assert len(p.export_state()["tombstones"]) == 2


def test_cap_drops_oldest_when_not_stale() -> None:
    p = MetricsProcessor(retention=10**9, max_sources=2)
    for i in range(5):
        p.process(
            parse_prometheus(metrics_text(req(LOCAL, "qwen-local", 1, client=f"c{i}"))), TOPO, now=i
        )
    assert p.source_count == 2


def test_roundtrip_with_unresolved_pending_and_tombstones() -> None:
    p = MetricsProcessor(retention=5)
    p.process(parse_prometheus(metrics_text(req(LOCAL, "qwen-local", 4))), TOPO, now=0)
    fail = metric("litellm_proxy_failed_requests_metric_total", "", 7, requested_model="g")
    p.process(parse_prometheus(metrics_text(fail)), Topology(), now=100)  # drops the request source
    p2 = MetricsProcessor()
    assert p2.load_state(_json_roundtrip(p))
    d = p2.process([], Topology({LOCAL: ModelInfo(LOCAL, "g")}), now=101)
    assert d.backends[LOCAL].failures == 7 and d.backends[LOCAL].requests == 4
    d = p2.process(parse_prometheus(metrics_text(req(LOCAL, "qwen-local", 6))), TOPO, now=102)
    assert d.backends[LOCAL].requests == 6  # tombstone continuation: 4 + 2


def test_live_series_over_caps_never_change_totals(monkeypatch) -> None:
    from custom_components.litellm_router import stats

    monkeypatch.setattr(stats, "MAX_TOMBSTONES", 2)
    p = MetricsProcessor(retention=10**9, max_sources=3)
    lines = [metric("litellm_input_tokens_metric_total", LOCAL, 100, k=i) for i in range(10)]
    totals = []
    for n in range(4):
        d = p.process(parse_prometheus(metrics_text(*lines)), TOPO, now=n)
        totals.append(d.backends[LOCAL].tokens_in)
    assert totals == [300] * 4
    assert p.source_count == 3
    # growth of an admitted series still counts
    lines[0] = metric("litellm_input_tokens_metric_total", LOCAL, 150, k=0)
    assert (
        p.process(parse_prometheus(metrics_text(*lines)), TOPO, now=9).backends[LOCAL].tokens_in
        == 350
    )


def _bounded(p, cap):
    for name in ("_last_seen", "_pending_attr", "_pending_cls", "_tombstones", "_clients"):
        assert len(getattr(p, name)) <= cap, name
    assert len(p._acc._last) <= cap and len(p._acc._total) <= cap


def test_high_cardinality_keeps_every_map_bounded(monkeypatch) -> None:
    from custom_components.litellm_router import stats

    monkeypatch.setattr(stats, "MAX_TOMBSTONES", 5)
    monkeypatch.setattr(stats, "MAX_CLIENTS", 3)
    p = MetricsProcessor(retention=10**9, max_sources=5)
    tok = lambda k: metric("litellm_input_tokens_metric_total", LOCAL, 100, k=k)  # noqa: E731
    reqs = [req(LOCAL, "qwen-local", 1, client=f"c{i}") for i in range(2000)]
    for n in range(3):
        d = p.process(
            parse_prometheus(metrics_text(*reqs, *[tok(i) for i in range(2000)])), TOPO, now=n
        )
        _bounded(p, 5)
    assert d.backends[LOCAL].tokens_in <= 500 and p._saturated
    assert len(d.clients) <= 4  # three clients plus "other"


def test_saturated_roundtrip_keeps_totals_and_new_is_baseline() -> None:
    p = MetricsProcessor(max_sources=1)
    tok = lambda k, v: metric("litellm_input_tokens_metric_total", LOCAL, v, k=k)  # noqa: E731
    run = lambda proc, now, *ls: proc.process(parse_prometheus(metrics_text(*ls)), TOPO, now=now)  # noqa: E731
    run(p, 0, tok("A", 10), tok("B", 100))  # B ignored
    p2 = MetricsProcessor(max_sources=1)
    assert p2.load_state(_json_roundtrip(p))
    assert run(p2, 1, tok("B", 100)).backends[LOCAL].tokens_in == 10  # A absent -> B baseline
    assert run(p2, 2, tok("B", 130)).backends[LOCAL].tokens_in == 40


def test_tombstone_eviction_makes_unknown_sources_baselines(monkeypatch) -> None:
    from custom_components.litellm_router import stats

    monkeypatch.setattr(stats, "MAX_TOMBSTONES", 2)
    p = MetricsProcessor(retention=1)
    tok = lambda k: metric("litellm_input_tokens_metric_total", LOCAL, 100, k=k)  # noqa: E731
    run = lambda proc, now, *ls: proc.process(parse_prometheus(metrics_text(*ls)), TOPO, now=now)  # noqa: E731
    for i, k in enumerate("ABCD"):
        d = run(p, i * 10, tok(k))
    assert d.backends[LOCAL].tokens_in == 400
    assert p._saturated
    p2 = MetricsProcessor(retention=1)
    assert p2.load_state(_json_roundtrip(p))
    assert run(p2, 100, tok("A")).backends[LOCAL].tokens_in == 400  # evicted A returns unchanged
    assert run(p, 100, tok("A")).backends[LOCAL].tokens_in == 400
