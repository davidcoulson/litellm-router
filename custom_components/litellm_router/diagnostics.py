"""Diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.core import HomeAssistant

from . import LiteLLMConfigEntry
from .const import CONF_API_KEY

TO_REDACT = {CONF_API_KEY, "Authorization"}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: LiteLLMConfigEntry
) -> dict[str, Any]:
    """Return diagnostics with the API key redacted."""
    runtime = entry.runtime_data
    return {
        "entry": async_redact_data(dict(entry.data), TO_REDACT),
        "options": dict(entry.options),
        "health": asdict(runtime.health.data) if runtime.health.data else None,
        "metrics": asdict(runtime.metrics.data) if runtime.metrics.data else None,
    }
