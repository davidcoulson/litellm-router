"""Base entities and device info."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity, DataUpdateCoordinator

from .api import Topology, normalize_url, sanitize_url
from .const import CONF_URL, DOMAIN
from .stats import backend_name


def proxy_device(entry: ConfigEntry) -> DeviceInfo:
    """Return device info for the proxy."""
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name="LiteLLM proxy",
        manufacturer="BerriAI",
        model="LiteLLM proxy",
        configuration_url=sanitize_url(normalize_url(entry.data[CONF_URL])),
    )


class ProxyEntity[C: DataUpdateCoordinator](CoordinatorEntity[C]):
    """Entity attached to the proxy device."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: C, entry: ConfigEntry, key: str) -> None:
        """Initialise the entity."""
        super().__init__(coordinator)
        self._attr_translation_key = key
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = proxy_device(entry)


class BackendEntity[C: DataUpdateCoordinator](CoordinatorEntity[C]):
    """Entity attached to one backend device."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: C, entry: ConfigEntry, model_id: str, key: str) -> None:
        """Initialise the entity."""
        super().__init__(coordinator)
        self.model_id = model_id
        self._entry = entry
        self._attr_translation_key = key
        self._attr_unique_id = f"{entry.entry_id}_{model_id}_{key}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry.entry_id}_{model_id}")},
            name=self._device_name(entry),
            manufacturer="LiteLLM",
            model="Backend",
            via_device=(DOMAIN, entry.entry_id),
        )

    def _device_name(self, entry: ConfigEntry) -> str:
        data = entry.runtime_data.health.data  # always set: first refresh precedes setup
        return backend_name(Topology(models=data.backends), self.model_id)
