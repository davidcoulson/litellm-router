"""Setup, entities, resets, diagnostics."""

from datetime import timedelta
from types import SimpleNamespace

import aiohttp
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.litellm_router.const import DOMAIN
from custom_components.litellm_router.diagnostics import async_get_config_entry_diagnostics

from .conftest import (
    API_KEY,
    FALLBACK,
    FAST,
    HEALTH,
    LOCAL,
    MODEL_INFO,
    URL,
    metric,
    metrics_text,
    mock_proxy,
    req,
)


def lat(mid, s, c):
    return [
        metric("litellm_llm_api_latency_metric_sum", mid, s),
        metric("litellm_llm_api_latency_metric_count", mid, c),
    ]


def state_of(hass, ent_reg, entry, unique_suffix):
    uid = f"{entry.entry_id}_{unique_suffix}"
    for e in ent_reg.entities.values():
        if e.unique_id == uid:
            return hass.states.get(e.entity_id)
    raise AssertionError(uid)


async def advance(hass, seconds):
    async_fire_time_changed(hass, dt_util.utcnow() + timedelta(seconds=seconds))
    await hass.async_block_till_done(wait_background_tasks=True)


async def test_entities_and_updates(hass, aioclient_mock, entry) -> None:
    mock_proxy(
        aioclient_mock,
        metrics_text(
            req(LOCAL, "qwen-local", 10),
            metric("litellm_input_tokens_metric_total", LOCAL, 7),
            *lat(LOCAL, 10, 10),
        ),
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    ent = er.async_get(hass)

    assert state_of(hass, ent, entry, "proxy_reachable").state == "on"
    assert state_of(hass, ent, entry, f"{LOCAL}_healthy").state == "on"
    assert state_of(hass, ent, entry, f"{FAST}_healthy").state == "off"
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "10"
    assert state_of(hass, ent, entry, f"{LOCAL}_tokens_in").state == "7"
    assert state_of(hass, ent, entry, f"{FALLBACK}_requests").state == "0"
    assert state_of(hass, ent, entry, f"{LOCAL}_avg_latency").state == "unknown"
    assert state_of(hass, ent, entry, "active_backend").state == "idle"
    assert state_of(hass, ent, entry, "fallback_requests").state == "0"
    assert state_of(hass, ent, entry, "requests_by_client").attributes["10.0.0.9"] == 10
    assert (
        state_of(hass, ent, entry, f"{LOCAL}_healthy").attributes["api_base"] == "http://strata/v1"
    )

    devs = dr.async_get(hass)
    assert (
        len([d for d in devs.devices.values() if any(i[0] == DOMAIN for i in d.identifiers)]) == 4
    )
    local_dev = devs.async_get_device({(DOMAIN, f"{entry.entry_id}_{LOCAL}")})
    assert local_dev.name == "qwen-local" and local_dev.via_device_id

    # traffic, including a fallback
    mock_proxy(
        aioclient_mock,
        metrics_text(
            req(LOCAL, "qwen-local", 16), req(FALLBACK, "qwen-local", 3), *lat(LOCAL, 13, 16)
        ),
    )
    await advance(hass, 31)
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "16"
    assert state_of(hass, ent, entry, "fallback_requests").state == "3"
    assert state_of(hass, ent, entry, "active_backend").state == "qwen-local"
    assert state_of(hass, ent, entry, f"{LOCAL}_avg_latency").state == "0.5"

    # container restart: counters drop
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 1)))
    await advance(hass, 65)
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "17"

    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["entry"]["api_key"] == "**REDACTED**"
    assert API_KEY not in str(diag)
    assert diag["health"]["reachable"] is True


async def test_new_backend_added_and_proxy_outage(hass, aioclient_mock, entry) -> None:
    health = {
        **HEALTH,
        "healthy_endpoints": HEALTH["healthy_endpoints"][:1],
        "unhealthy_endpoints": [],
    }
    mock_proxy(aioclient_mock, metrics_text(), health=health, model_info={"json": {"data": []}})
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    ent = er.async_get(hass)
    assert not [e for e in ent.entities.values() if FALLBACK in e.unique_id]

    mock_proxy(aioclient_mock, metrics_text())  # full topology now
    await advance(hass, 65)
    assert state_of(hass, ent, entry, f"{FALLBACK}_healthy").state == "on"

    # proxy goes down: reachable off, backends unavailable
    err = {"exc": aiohttp.ClientError()}
    mock_proxy(aioclient_mock, liveliness=err, metrics_ep=err)
    await advance(hass, 31)
    await advance(hass, 65)
    assert state_of(hass, ent, entry, "proxy_reachable").state == "off"
    assert state_of(hass, ent, entry, f"{LOCAL}_healthy").state == "unavailable"
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "unavailable"

    # backend removed from /health while the proxy is up
    mock_proxy(
        aioclient_mock,
        metrics_text(),
        health={**HEALTH, "unhealthy_endpoints": [], "healthy_endpoints": []},
        model_info={"json": {"data": []}},
    )
    await advance(hass, 65)
    assert state_of(hass, ent, entry, f"{LOCAL}_healthy").state == "unavailable"
    assert state_of(hass, ent, entry, "proxy_reachable").state == "on"


async def test_setup_not_ready_and_auth_failure(hass, aioclient_mock, entry) -> None:
    mock_proxy(aioclient_mock, metrics_status=500)
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    assert entry.state is ConfigEntryState.SETUP_RETRY

    await hass.config_entries.async_unload(entry.entry_id)
    mock_proxy(aioclient_mock, health_ep={"status": 401})
    await hass.config_entries.async_setup(entry.entry_id)
    assert entry.state is ConfigEntryState.SETUP_ERROR
    assert any(
        f["context"]["source"] == "reauth" for f in hass.config_entries.flow.async_progress()
    )

    # liveliness auth failure
    mock_proxy(aioclient_mock, liveliness={"status": 401})
    entry2 = entry.__class__(domain=DOMAIN, data=dict(entry.data), unique_id="other", title="x")
    entry2.add_to_hass(hass)
    await hass.config_entries.async_setup(entry2.entry_id)
    assert entry2.state is ConfigEntryState.SETUP_ERROR


async def test_metrics_failures(hass, aioclient_mock, entry) -> None:
    mock_proxy(aioclient_mock, metrics_text())
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    mock_proxy(aioclient_mock, metrics_status=401)
    await advance(hass, 31)
    assert any(
        f["context"]["source"] == "reauth" for f in hass.config_entries.flow.async_progress()
    )
    mock_proxy(aioclient_mock, metrics_status=500)
    await advance(hass, 31)
    assert not entry.runtime_data.metrics.last_update_success
    assert await hass.config_entries.async_unload(entry.entry_id)


async def test_settings_outage_keeps_fallbacks_and_backend_membership(
    hass, aioclient_mock, entry
) -> None:
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 1)))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    ent = er.async_get(hass)
    health = entry.runtime_data.health
    assert health.data.topology.fallbacks == {"qwen-local": ["qwen-fallback"]}

    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 1)), router={"status": 500})
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert health.data.topology.fallbacks == {"qwen-local": ["qwen-fallback"]}

    # backend vanishes from /health and /model/info while metrics keep working
    mock_proxy(
        aioclient_mock,
        metrics_text(req(LOCAL, "qwen-local", 1)),
        health={
            **HEALTH,
            "healthy_endpoints": HEALTH["healthy_endpoints"][1:],
            "unhealthy_endpoints": [],
        },
        model_info={"json": {"data": []}},
    )
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "unavailable"
    assert state_of(hass, ent, entry, f"{FALLBACK}_requests").state != "unavailable"

    # it returns and resumes with its accumulated counters
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 3)))
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "3"


async def test_liveliness_ok_while_health_times_out(hass, aioclient_mock, entry) -> None:
    mock_proxy(aioclient_mock, metrics_text())
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    ent = er.async_get(hass)

    mock_proxy(aioclient_mock, metrics_text(), liveliness={"exc": aiohttp.ClientError()})
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert state_of(hass, ent, entry, "proxy_reachable").state == "off"

    # alive again, but /health times out: reachable recovers, backends blank
    for _ in range(2):
        mock_proxy(aioclient_mock, metrics_text(), health_ep={"exc": TimeoutError()})
        await advance(hass, 65)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert state_of(hass, ent, entry, "proxy_reachable").state == "on"
        assert state_of(hass, ent, entry, f"{LOCAL}_healthy").state == "unavailable"
        assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "unavailable"

    mock_proxy(aioclient_mock, metrics_text())
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert state_of(hass, ent, entry, f"{LOCAL}_healthy").state == "on"


async def test_setup_with_health_timing_out(hass, aioclient_mock, entry) -> None:
    mock_proxy(
        aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 4)), health_ep={"exc": TimeoutError()}
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.state is ConfigEntryState.LOADED
    ent = er.async_get(hass)
    assert state_of(hass, ent, entry, "proxy_reachable").state == "on"
    assert state_of(hass, ent, entry, "active_backend").state == "idle"

    # health recovers: backends appear
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 4)))
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert state_of(hass, ent, entry, f"{LOCAL}_healthy").state == "on"
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "4"


async def test_totals_survive_reload_and_pending_too(
    hass, aioclient_mock, entry, hass_storage
) -> None:
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 100)))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    ent = er.async_get(hass)
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 3)))  # proxy restarted
    await advance(hass, 31)
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "103"

    # reload with unchanged metrics: totals must not drop back to 3
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert state_of(hass, ent, entry, f"{LOCAL}_requests").state == "103"
    assert f"{DOMAIN}.{entry.entry_id}" in hass_storage

    # removing the entry deletes the stored totals
    await hass.config_entries.async_remove(entry.entry_id)
    assert f"{DOMAIN}.{entry.entry_id}" not in hass_storage


async def test_pending_growth_survives_reload(hass, aioclient_mock, entry) -> None:
    mock_proxy(
        aioclient_mock,
        metrics_text(req(FALLBACK, "qwen-local", 5)),
        model_info={"json": {"data": []}},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    ent = er.async_get(hass)
    assert state_of(hass, ent, entry, "fallback_requests").state == "0"

    mock_proxy(aioclient_mock, metrics_text(req(FALLBACK, "qwen-local", 5)))  # topology back
    await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert state_of(hass, ent, entry, "fallback_requests").state == "5"


async def test_urls_are_sanitized_in_diagnostics_and_attributes(hass, aioclient_mock) -> None:
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    secret_url = "http://admin:hunter2@litellm.test:4000"
    e = MockConfigEntry(
        domain=DOMAIN,
        unique_id="http://litellm.test:4000",
        data={"url": secret_url, "api_key": API_KEY},
    )
    health = {
        **HEALTH,
        "healthy_endpoints": [
            {
                "api_base": "http://u:pw@strata/v1?api_key=SEKRET",
                "model": "openai/qwen",
                "model_id": LOCAL,
            }
        ],
        "unhealthy_endpoints": [],
    }
    for path in ("/health/liveliness", "/metrics/"):
        aioclient_mock.get(f"{secret_url}{path}", text="")
    aioclient_mock.get(f"{secret_url}/health", json=health)
    aioclient_mock.get(f"{secret_url}/model/info", json={"data": []})
    aioclient_mock.get(f"{secret_url}/router/settings", json={})
    e.add_to_hass(hass)
    assert await hass.config_entries.async_setup(e.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)

    diag = str(await async_get_config_entry_diagnostics(hass, e))
    for leak in ("hunter2", "admin", "pw@", "SEKRET", API_KEY):
        assert leak not in diag
    ent = er.async_get(hass)
    st = state_of(hass, ent, e, f"{LOCAL}_healthy")
    assert st.attributes["api_base"] == "http://strata/v1?api_key=**REDACTED**"
    dev = dr.async_get(hass).async_get_device({(DOMAIN, e.entry_id)})
    assert dev.configuration_url == "http://litellm.test:4000"


async def test_checkpoints_written_during_fast_polling(hass, aioclient_mock, hass_storage) -> None:
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    entry = MockConfigEntry(
        domain=DOMAIN,
        unique_id=URL,
        data={"url": URL, "api_key": API_KEY},
        options={"metrics_interval": 10, "health_interval": 10},
    )
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 5)))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    key = f"{DOMAIN}.{entry.entry_id}"
    assert key not in hass_storage
    base = dt_util.utcnow()
    for i in range(1, 8):  # 70 s of continuous 10 s polling
        async_fire_time_changed(hass, base + timedelta(seconds=10 * i))
        await hass.async_block_till_done(wait_background_tasks=True)
    assert hass_storage[key]["data"]["attr"]
    await hass.config_entries.async_unload(entry.entry_id)


async def test_remove_orphan_backend_device(hass, aioclient_mock, entry) -> None:
    from types import SimpleNamespace

    from custom_components.litellm_router import async_remove_config_entry_device

    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 5), req(FAST, "qwen-fast", 7)))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    ent = er.async_get(hass)
    devs = dr.async_get(hass)
    proxy = devs.async_get_device({(DOMAIN, entry.entry_id)})
    local = devs.async_get_device({(DOMAIN, f"{entry.entry_id}_{LOCAL}")})
    fast = devs.async_get_device({(DOMAIN, f"{entry.entry_id}_{FAST}")})
    assert not await async_remove_config_entry_device(hass, entry, proxy)
    assert not await async_remove_config_entry_device(hass, entry, local)
    assert not await async_remove_config_entry_device(hass, entry, fast)  # still configured
    assert state_of(hass, ent, entry, f"{FAST}_requests").state == "7"

    # FAST is removed from the proxy
    health = {**HEALTH, "unhealthy_endpoints": []}
    mock_proxy(
        aioclient_mock,
        metrics_text(req(LOCAL, "qwen-local", 5)),
        health=health,
        model_info={"json": {"data": MODEL_INFO["data"][:2]}},
    )
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert await async_remove_config_entry_device(hass, entry, fast)
    assert FAST not in str(entry.runtime_data.metrics.processor.export_state())
    assert LOCAL in str(entry.runtime_data.metrics.processor.export_state())

    # not loaded / no data yet: allowed for backend devices, still refused for the proxy
    foreign = SimpleNamespace(identifiers={(DOMAIN, f"{entry.entry_id}_{FAST}")})
    unloaded = SimpleNamespace(entry_id=entry.entry_id)
    assert await async_remove_config_entry_device(hass, unloaded, foreign)
    nodata = SimpleNamespace(
        entry_id=entry.entry_id,
        runtime_data=SimpleNamespace(
            known={},
            health=SimpleNamespace(data=None),
            metrics=SimpleNamespace(
                processor=entry.runtime_data.metrics.processor,
                async_save_state=entry.runtime_data.metrics.async_save_state,
            ),
        ),
    )
    assert await async_remove_config_entry_device(hass, nodata, foreign)
    assert not await async_remove_config_entry_device(
        hass, unloaded, SimpleNamespace(identifiers={("other", "x")})
    )


async def test_returning_backend_recreates_entities_after_device_removal(
    hass, aioclient_mock, entry
) -> None:
    from custom_components.litellm_router import async_remove_config_entry_device

    mock_proxy(aioclient_mock, metrics_text(req(FAST, "qwen-fast", 7)))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    ent, devs = er.async_get(hass), dr.async_get(hass)
    fast_ident = {(DOMAIN, f"{entry.entry_id}_{FAST}")}

    mock_proxy(
        aioclient_mock,
        metrics_text(),
        health={**HEALTH, "unhealthy_endpoints": []},
        model_info={"json": {"data": MODEL_INFO["data"][:2]}},
    )
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    device = devs.async_get_device(fast_ident)
    assert await async_remove_config_entry_device(hass, entry, device)
    devs.async_remove_device(device.id)  # real registry removal (removes its entities)
    await hass.async_block_till_done()
    assert devs.async_get_device(fast_ident) is None
    assert not [e for e in ent.entities.values() if FAST in e.unique_id]

    mock_proxy(aioclient_mock, metrics_text())  # backend is back
    await advance(hass, 65)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert devs.async_get_device(fast_ident) is not None
    assert state_of(hass, ent, entry, f"{FAST}_healthy").state == "off"
    assert state_of(hass, ent, entry, f"{FAST}_requests").state == "0"


async def test_delete_while_unloaded_cleans_store(
    hass, aioclient_mock, entry, hass_storage
) -> None:
    from custom_components.litellm_router import async_remove_config_entry_device

    mock_proxy(aioclient_mock, metrics_text(req(FAST, "qwen-fast", 7), req(LOCAL, "qwen-local", 2)))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert await hass.config_entries.async_unload(entry.entry_id)
    key = f"{DOMAIN}.{entry.entry_id}"
    assert FAST in str(hass_storage[key])

    device = SimpleNamespace(identifiers={(DOMAIN, f"{entry.entry_id}_{FAST}")})
    assert await async_remove_config_entry_device(hass, entry, device)
    assert FAST not in str(hass_storage[key]) and LOCAL in str(hass_storage[key])

    # no stored state at all: nothing to clean, still allowed
    del hass_storage[key]
    assert await async_remove_config_entry_device(hass, entry, device)
    assert key not in hass_storage

    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 2)))
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert FAST not in str(entry.runtime_data.metrics.processor.export_state())
