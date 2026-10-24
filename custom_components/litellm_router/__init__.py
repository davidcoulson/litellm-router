"""The LiteLLM Router integration."""

from __future__ import annotations

from dataclasses import dataclass

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import LiteLLMClient
from .const import CONF_API_KEY, CONF_URL
from .coordinator import HealthCoordinator, MetricsCoordinator

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR]


@dataclass(slots=True)
class LiteLLMRuntimeData:
    """Runtime objects stored on the config entry."""

    client: LiteLLMClient
    health: HealthCoordinator
    metrics: MetricsCoordinator


type LiteLLMConfigEntry = ConfigEntry[LiteLLMRuntimeData]


async def async_setup_entry(hass: HomeAssistant, entry: LiteLLMConfigEntry) -> bool:
    """Set up a proxy from a config entry."""
    client = LiteLLMClient(
        async_get_clientsession(hass), entry.data[CONF_URL], entry.data[CONF_API_KEY]
    )
    health = HealthCoordinator(hass, entry, client)
    # The first refresh raises ConfigEntryNotReady / ConfigEntryAuthFailed as needed.
    await health.async_config_entry_first_refresh()
    metrics = MetricsCoordinator(hass, entry, client, health)
    await metrics.async_config_entry_first_refresh()
    entry.runtime_data = LiteLLMRuntimeData(client, health, metrics)
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _async_reload(hass: HomeAssistant, entry: LiteLLMConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LiteLLMConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
