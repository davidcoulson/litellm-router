"""The LiteLLM Router integration."""

from __future__ import annotations

from dataclasses import dataclass, field

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntry

from .api import LiteLLMClient
from .const import CONF_API_KEY, CONF_URL, DOMAIN
from .coordinator import HealthCoordinator, MetricsCoordinator, metrics_store
from .stats import MetricsProcessor

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SENSOR]


@dataclass(slots=True)
class LiteLLMRuntimeData:
    """Runtime objects stored on the config entry."""

    client: LiteLLMClient
    health: HealthCoordinator
    metrics: MetricsCoordinator
    # model_ids that already have entities, per platform (cleared on device removal)
    known: dict[str, set[str]] = field(default_factory=lambda: {p: set() for p in PLATFORMS})


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
    await metrics.async_load_state()
    metrics.start_checkpoints()
    await metrics.async_config_entry_first_refresh()
    entry.runtime_data = LiteLLMRuntimeData(client, health, metrics)
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _async_reload(hass: HomeAssistant, entry: LiteLLMConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: LiteLLMConfigEntry) -> bool:
    """Unload a config entry."""
    await entry.runtime_data.metrics.async_save_state()
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_remove_entry(hass: HomeAssistant, entry: LiteLLMConfigEntry) -> None:
    """Delete the stored totals when the entry is removed."""
    await metrics_store(hass, entry.entry_id).async_remove()


async def async_remove_config_entry_device(
    hass: HomeAssistant, entry: LiteLLMConfigEntry, device_entry: DeviceEntry
) -> bool:
    """Allow deleting devices of backends that no longer exist (never the proxy)."""
    prefix = f"{entry.entry_id}_"
    model_ids = {
        ident[len(prefix) :]
        for domain, ident in device_entry.identifiers
        if domain == DOMAIN and ident.startswith(prefix)
    }
    if not model_ids:
        return False  # the proxy device (or not ours)
    runtime = getattr(entry, "runtime_data", None)
    data = runtime.health.data if runtime else None
    if data is not None and model_ids & set(data.backends):
        return False
    if runtime:
        for model_id in model_ids:
            runtime.metrics.processor.forget_backend(model_id)
            for known in runtime.known.values():
                known.discard(model_id)
        await runtime.metrics.async_save_state()
    else:
        # not loaded: clean the stored totals so they don't return on the next setup
        store = metrics_store(hass, entry.entry_id)
        processor = MetricsProcessor()
        if processor.load_state(await store.async_load()):
            for model_id in model_ids:
                processor.forget_backend(model_id)
            await store.async_save(processor.export_state())
    return True
