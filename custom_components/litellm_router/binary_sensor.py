"""Binary sensors."""

from __future__ import annotations

from homeassistant.components.binary_sensor import BinarySensorDeviceClass, BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import LiteLLMConfigEntry
from .api import sanitize_url
from .coordinator import HealthCoordinator
from .entity import BackendEntity, ProxyEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LiteLLMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up binary sensors, adding backends as they appear."""
    health = entry.runtime_data.health
    async_add_entities([ProxyReachable(health, entry, "proxy_reachable")])
    known: set[str] = set()

    def _add_new() -> None:
        new = set(health.data.backends) - known
        known.update(new)
        async_add_entities(BackendHealthy(health, entry, model_id, "healthy") for model_id in new)

    _add_new()
    entry.async_on_unload(health.async_add_listener(_add_new))


class ProxyReachable(ProxyEntity[HealthCoordinator], BinarySensorEntity):
    """The proxy answers /health/liveliness."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY

    @property
    def is_on(self) -> bool:
        """Return True when reachable."""
        return self.coordinator.data.reachable

    @property
    def available(self) -> bool:
        """Stay available so that an outage is visible."""
        return self.coordinator.data is not None


class BackendHealthy(BackendEntity[HealthCoordinator], BinarySensorEntity):
    """Backend healthy per /health."""

    _attr_device_class = BinarySensorDeviceClass.CONNECTIVITY

    @property
    def available(self) -> bool:
        """Unavailable while the proxy is down or the backend is no longer reported."""
        data = self.coordinator.data
        return (
            super().available
            and data.reachable
            and data.backends_ok
            and (self.model_id in data.health.healthy or self.model_id in data.health.unhealthy)
        )

    @property
    def is_on(self) -> bool:
        """Return True when the backend is healthy."""
        return self.model_id in self.coordinator.data.health.healthy

    @property
    def extra_state_attributes(self) -> dict[str, str]:
        """Expose where the backend lives."""
        info = self.coordinator.data.backends.get(self.model_id)
        return {"api_base": sanitize_url(info.api_base), "model": info.model} if info else {}
