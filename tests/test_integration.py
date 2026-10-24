"""Setup, entities, resets, diagnostics."""

from datetime import timedelta

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
    mock_proxy(aioclient_mock, health_ep={"status": 500})
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
