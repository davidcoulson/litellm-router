"""litellm_router.reload_code: unload, re-import the modules from disk, set up again."""

import sys

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.helpers import issue_registry as ir

from custom_components import litellm_router
from custom_components.litellm_router.const import DOMAIN

from .conftest import LOCAL, metrics_text, mock_proxy, req

PACKAGE = "custom_components.litellm_router"


@pytest.fixture(autouse=True)
def keep_original_modules():
    """Later tests import (and patch) the package by name: put the original modules back."""
    saved = {n: m for n, m in sys.modules.items() if n.startswith(PACKAGE)}
    yield
    for name in [n for n in sys.modules if n.startswith(PACKAGE)]:
        del sys.modules[name]
    sys.modules.update(saved)


async def test_reload_code_replaces_the_modules(hass, aioclient_mock, entry) -> None:
    mock_proxy(aioclient_mock, metrics_text(req(LOCAL, "qwen-local", 4)))
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.state is ConfigEntryState.LOADED
    old = {n: m for n, m in sys.modules.items() if n.startswith(PACKAGE)}

    await hass.services.async_call(DOMAIN, "reload_code", {}, blocking=True)
    await hass.async_block_till_done(wait_background_tasks=True)

    assert entry.state is ConfigEntryState.LOADED
    assert sys.modules[PACKAGE] is not old[PACKAGE]
    assert hass.services.has_service(DOMAIN, "reload_code")
    assert type(entry.runtime_data) is sys.modules[PACKAGE].LiteLLMRuntimeData
    assert type(entry.runtime_data) is not old[PACKAGE].LiteLLMRuntimeData


async def test_reload_clears_only_marketplaces_litellm_router_restart_repairs(hass):
    for domain, issue_id, issue_domain in (
        ("marketplace", "restart_required_1_tags/2026.11.27.2", "litellm_router"),
        ("marketplace", "restart_required_2_tags/2026.11.26.01", "ssh"),
        ("marketplace", "critical_x", "litellm_router"),
        ("litellm_router", "restart_required_x", "litellm_router"),
    ):
        ir.async_create_issue(
            hass, domain, issue_id, is_fixable=False, issue_domain=issue_domain,
            severity=ir.IssueSeverity.WARNING, translation_key="x",
        )
    litellm_router._clear_marketplace_restart(hass)
    assert set(ir.async_get(hass).issues) == {
        ("marketplace", "restart_required_2_tags/2026.11.26.01"),
        ("marketplace", "critical_x"),
        ("litellm_router", "restart_required_x"),
    }
