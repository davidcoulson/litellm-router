"""Prometheus parser and counter accumulator."""

from custom_components.litellm_router.prometheus import parse_prometheus
from custom_components.litellm_router.tracker import CounterAccumulator


def test_parse_prometheus() -> None:
    text = "\n".join(
        [
            "# HELP a b",
            "",
            'm_total{a="1",b="x, \\"y\\" } {z}\\\\\\n",} 3.0',
            "plain 5 1700000000",
            "nanval NaN",
            "inf_val +Inf",
            "noval",
            'broken{a="1" 4',
            "broken2{a=1} 4",
            'unterminated{a="1',
            'noclose{a="1"',
            "badnum 1.2.3",
            '{a="1"} 4',
            "empty{} 2",
        ]
    )
    samples = {s.name: s for s in parse_prometheus(text)}
    assert samples["m_total"].labels == {"a": "1", "b": 'x, "y" } {z}\\\n'}
    assert samples["m_total"].value == 3.0
    assert samples["plain"].value == 5
    assert samples["inf_val"].value == float("inf")
    assert samples["empty"].labels == {}
    assert "nanval" not in samples and "noval" not in samples
    assert not {"broken", "broken2", "unterminated", "badnum", "noclose"} & samples.keys()


def test_accumulator_baseline_new_reset_and_return() -> None:
    acc = CounterAccumulator()
    assert not acc.primed
    assert acc.update("a", 10) == 0  # baseline
    acc.finish_poll()
    assert acc.primed
    assert acc.total("a") == 10
    assert acc.update("a", 14) == 4
    assert acc.update("new", 3) == 3  # series born after baseline counts fully
    assert acc.total("new") == 3
    assert acc.update("a", 2) == 2  # restart: never negative
    assert acc.total("a") == 16
    # series absent for a while, then back lower (restart meanwhile)
    assert acc.update("a", 1) == 1
    assert acc.update("a", 1) == 0
    assert acc.total("a") == 17
    assert acc.total("missing") == 0
    assert set(acc.series()) == {"a", "new"}


def test_accumulator_restore_forget() -> None:
    acc = CounterAccumulator()
    acc.restore("a", 5.0, 9.0)
    acc.set_primed(True)
    assert acc.primed and acc.last("a") == 5.0 and acc.total("a") == 9.0
    assert acc.update("a", 7) == 2
    acc.forget("a")
    acc.forget("a")
    assert acc.last("a") is None and acc.total("a") == 0
