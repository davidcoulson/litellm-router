"""Config, reauth, options flows."""

import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from custom_components.litellm_router.const import DOMAIN

from .conftest import API_KEY, URL, mock_proxy


async def test_user_flow(hass, aioclient_mock) -> None:
    mock_proxy(aioclient_mock)
    r = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    assert r["type"] is FlowResultType.FORM
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"], {"url": "litellm.test:4000/", "api_key": API_KEY}
    )
    # bare host gets http:// and the entry is created
    assert r["type"] is FlowResultType.CREATE_ENTRY
    assert r["data"] == {"url": URL, "api_key": API_KEY}
    assert r["result"].unique_id == URL


@pytest.mark.parametrize(("status", "error"), [(401, "invalid_auth"), (500, "cannot_connect")])
async def test_user_flow_errors_then_success(hass, aioclient_mock, status, error) -> None:
    mock_proxy(aioclient_mock, metrics_status=status)
    r = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"], {"url": URL, "api_key": API_KEY}
    )
    assert r["errors"] == {"base": error}
    mock_proxy(aioclient_mock)
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"], {"url": URL, "api_key": API_KEY}
    )
    assert r["type"] is FlowResultType.CREATE_ENTRY


async def test_duplicate(hass, aioclient_mock, entry) -> None:
    entry.add_to_hass(hass)
    r = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"], {"url": URL, "api_key": API_KEY}
    )
    assert r["type"] is FlowResultType.ABORT and r["reason"] == "already_configured"


async def test_reauth(hass, aioclient_mock, entry) -> None:
    entry.add_to_hass(hass)
    mock_proxy(aioclient_mock, metrics_status=401)
    r = await entry.start_reauth_flow(hass)
    assert r["step_id"] == "reauth_confirm"
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"api_key": "bad"})
    assert r["errors"] == {"base": "invalid_auth"}
    mock_proxy(aioclient_mock)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"api_key": "sk-new"})
    assert r["type"] is FlowResultType.ABORT and r["reason"] == "reauth_successful"
    assert entry.data["api_key"] == "sk-new"
    await hass.async_block_till_done(wait_background_tasks=True)
    await hass.config_entries.async_unload(entry.entry_id)


async def test_options(hass, aioclient_mock, entry) -> None:
    entry.add_to_hass(hass)
    mock_proxy(aioclient_mock)
    assert await hass.config_entries.async_setup(entry.entry_id)
    r = await hass.config_entries.options.async_init(entry.entry_id)
    assert r["type"] is FlowResultType.FORM
    r = await hass.config_entries.options.async_configure(
        r["flow_id"], {"health_interval": 120, "metrics_interval": 15}
    )
    assert r["type"] is FlowResultType.CREATE_ENTRY
    await hass.async_block_till_done()
    assert entry.state is config_entries.ConfigEntryState.LOADED
    assert entry.runtime_data.health.update_interval.total_seconds() == 120
    assert entry.runtime_data.metrics.update_interval.total_seconds() == 15
    # reopening shows the stored values
    r = await hass.config_entries.options.async_init(entry.entry_id)
    assert r["type"] is FlowResultType.FORM


async def test_reconfigure(hass, aioclient_mock, entry) -> None:
    entry.add_to_hass(hass)
    mock_proxy(aioclient_mock)
    assert await hass.config_entries.async_setup(entry.entry_id)
    r = await entry.start_reconfigure_flow(hass)
    assert r["step_id"] == "reconfigure"
    new = "http://new.test:4000"

    def register(metrics_status=200):
        aioclient_mock.clear_requests()
        aioclient_mock.get(f"{new}/health/liveliness", text="ok")
        aioclient_mock.get(f"{new}/metrics/", status=metrics_status, text="")
        aioclient_mock.get(f"{new}/health", json={})
        aioclient_mock.get(f"{new}/model/info", json={})
        aioclient_mock.get(f"{new}/router/settings", json={})

    register(500)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"url": new})
    assert r["errors"] == {"base": "cannot_connect"}
    register()
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"url": "new.test:4000/"})
    assert r["type"] is FlowResultType.ABORT and r["reason"] == "reconfigure_successful"
    assert entry.data == {"url": new, "api_key": API_KEY}
    assert entry.unique_id == new
    await hass.async_block_till_done()


async def test_reconfigure_duplicate_and_id_moves(hass, aioclient_mock, entry) -> None:
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    other_url = "http://other.test:4000"
    other = MockConfigEntry(
        domain=DOMAIN, unique_id=other_url, data={"url": other_url, "api_key": API_KEY}
    )
    other.add_to_hass(hass)
    entry.add_to_hass(hass)
    mock_proxy(aioclient_mock)
    r = await entry.start_reconfigure_flow(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"url": other_url + "/"})
    assert r["type"] is FlowResultType.ABORT and r["reason"] == "already_configured"

    # move entry from URL to a new address (no setup, so no reload side effects)
    new = "http://moved.test:4000"
    for path in ("/health/liveliness", "/metrics/"):
        aioclient_mock.get(f"{new}{path}", text="")
    r = await entry.start_reconfigure_flow(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"url": new})
    assert r["reason"] == "reconfigure_successful"
    assert entry.unique_id == new
    await hass.async_block_till_done()

    # old URL can now be added; the new one is blocked
    r = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"], {"url": URL, "api_key": API_KEY}
    )
    assert r["type"] is FlowResultType.CREATE_ENTRY
    r = await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"], {"url": new, "api_key": API_KEY}
    )
    assert r["type"] is FlowResultType.ABORT


def _gated_validate(monkeypatch, n):
    """Make _validate wait until n callers are inside it, then succeed."""
    import asyncio

    from custom_components.litellm_router import config_flow

    inside = []
    release = asyncio.Event()

    async def fake(hass, url, key):
        inside.append(url)
        if len(inside) >= n:
            release.set()
        await release.wait()

    monkeypatch.setattr(config_flow, "_validate", fake)


async def test_concurrent_user_flows_same_url(hass, monkeypatch) -> None:
    import asyncio

    _gated_validate(monkeypatch, 2)
    flows = [
        await hass.config_entries.flow.async_init(DOMAIN, context={"source": "user"})
        for _ in range(2)
    ]
    results = await asyncio.gather(
        *(
            hass.config_entries.flow.async_configure(f["flow_id"], {"url": URL, "api_key": API_KEY})
            for f in flows
        )
    )
    assert sorted(r["type"] for r in results) == sorted(
        [FlowResultType.CREATE_ENTRY, FlowResultType.ABORT]
    )
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1


async def test_concurrent_reconfigure_same_url(hass, monkeypatch) -> None:
    import asyncio

    from pytest_homeassistant_custom_component.common import MockConfigEntry

    entries = []
    for i in range(2):
        e = MockConfigEntry(
            domain=DOMAIN,
            unique_id=f"http://p{i}:1",
            data={"url": f"http://p{i}:1", "api_key": API_KEY},
        )
        e.add_to_hass(hass)
        entries.append(e)
    _gated_validate(monkeypatch, 2)
    flows = [await e.start_reconfigure_flow(hass) for e in entries]
    results = await asyncio.gather(
        *(
            hass.config_entries.flow.async_configure(f["flow_id"], {"url": "http://same:1"})
            for f in flows
        )
    )
    assert sorted(r["reason"] for r in results) == ["already_configured", "reconfigure_successful"]
    assert sorted(e.unique_id for e in entries).count("http://same:1") == 1
    await hass.async_block_till_done()
