"""The LiteLLM Router integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import importlib
import sys

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import Event, HomeAssistant, ServiceCall, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntry
from homeassistant.helpers.service import async_register_admin_service
from homeassistant.loader import async_get_integration

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
    # platform entity-discovery callbacks, re-run after a backend device is really removed
    # model_ids that already have entities, per platform (forgotten once the device is gone)
    known: dict[str, set[str]] = field(default_factory=lambda: {p: set() for p in PLATFORMS})
    rediscover: list[Callable[[], None]] = field(default_factory=list)


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
    runtime = LiteLLMRuntimeData(client, health, metrics)
    entry.runtime_data = runtime

    @callback
    def _device_registry_updated(event: Event) -> None:
        # Only once a device is really gone can its entities be recreated for a returning backend.
        if event.data["action"] != "remove":
            return
        # Iterate instead of async_get_device_by_identifier, which needs newer HA than hacs.json allows.
        identifiers = {
            i
            for d in dr.async_entries_for_config_entry(dr.async_get(hass), entry.entry_id)
            for i in d.identifiers
        }
        for known in runtime.known.values():
            for model_id in list(known):
                if (DOMAIN, f"{entry.entry_id}_{model_id}") not in identifiers:
                    known.discard(model_id)
        for discover in runtime.rediscover:
            discover()

    entry.async_on_unload(hass.bus.async_listen(dr.EVENT_DEVICE_REGISTRY_UPDATED, _device_registry_updated))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    _register_reload_code(hass)
    return True


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


def _forget_code(hass: HomeAssistant, package: str, domain: str) -> None:
    """Drop a package's modules so the next setup imports them from disk again.

    Home Assistant keeps its own copy of each imported integration module
    (hass.data["components"], keyed "<domain>" and "<domain>.<platform>") next to
    sys.modules; both have to go.
    """
    for cache, root in ((sys.modules, package), (hass.data.get("components", {}), domain),
                        (hass.data.get("missing_platforms", {}), domain)):
        for key in [n for n in cache if n == root or n.startswith(root + ".")]:
            del cache[key]
    importlib.invalidate_caches()  # or a file added since the last import may not be found


def _register_reload_code(hass: HomeAssistant) -> None:
    """Dev: litellm_router.reload_code loads the Python files from disk without a restart."""

    async def _reload_code(call: ServiceCall) -> None:
        entries = hass.config_entries.async_entries(DOMAIN)
        for entry in entries:
            if not await hass.config_entries.async_unload(entry.entry_id):
                raise HomeAssistantError(f"{DOMAIN} did not unload; restart Home Assistant instead")
        _forget_code(hass, __name__, DOMAIN)
        try:
            await (await async_get_integration(hass, DOMAIN)).async_get_component()
        except Exception as err:
            # This handler stays registered, so the action works again once the code is fixed.
            raise HomeAssistantError(f"{DOMAIN} is unloaded: the new code did not load ({err}). "
                                     f"Fix it and call {DOMAIN}.reload_code again") from err
        failed = [e.title for e in entries if not await hass.config_entries.async_setup(e.entry_id)]
        if failed:
            raise HomeAssistantError(f"{DOMAIN} did not set up again for {failed}; see the log")

    async_register_admin_service(hass, DOMAIN, "reload_code", _reload_code)
