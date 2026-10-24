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
