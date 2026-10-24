"""Sensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.const import EntityCategory, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import LiteLLMConfigEntry
from .coordinator import MetricsCoordinator
from .entity import BackendEntity, ProxyEntity
from .stats import BackendStats

PARALLEL_UPDATES = 0


@dataclass(frozen=True, kw_only=True)
class BackendSensorDescription(SensorEntityDescription):
    """Description of a per-backend sensor."""

    value_fn: Callable[[BackendStats], float | None]


BACKEND_SENSORS: tuple[BackendSensorDescription, ...] = (
    BackendSensorDescription(
        key="requests",
        translation_key="requests",
        state_class=SensorStateClass.TOTAL_INCREASING,
        value_fn=lambda s: int(s.requests),
    ),
    BackendSensorDescription(
        key="failures",
        translation_key="failures",
        state_class=SensorStateClass.TOTAL_INCREASING,
        value_fn=lambda s: int(s.failures),
    ),
    BackendSensorDescription(
        key="tokens_in",
        translation_key="tokens_in",
        state_class=SensorStateClass.TOTAL_INCREASING,
        value_fn=lambda s: int(s.tokens_in),
    ),
    BackendSensorDescription(
        key="tokens_out",
        translation_key="tokens_out",
        state_class=SensorStateClass.TOTAL_INCREASING,
        value_fn=lambda s: int(s.tokens_out),
    ),
    BackendSensorDescription(
        key="avg_latency",
        translation_key="avg_latency",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=2,
        value_fn=lambda s: s.avg_latency,
    ),
    BackendSensorDescription(
        key="avg_ttft",
        translation_key="avg_ttft",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.SECONDS,
        state_class=SensorStateClass.MEASUREMENT,
        suggested_display_precision=2,
        value_fn=lambda s: s.avg_ttft,
    ),
)

_EMPTY = BackendStats()


async def async_setup_entry(
    hass: HomeAssistant,
    entry: LiteLLMConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up sensors, adding backends as they appear."""
    runtime = entry.runtime_data
    async_add_entities(
        [
            ActiveBackend(runtime.metrics, entry, "active_backend"),
            FallbackRequests(runtime.metrics, entry, "fallback_requests"),
            RequestsByClient(runtime.metrics, entry, "requests_by_client"),
        ]
    )
    known: set[str] = set()

    def _add_new() -> None:
        new = set(runtime.health.data.backends) - known
        known.update(new)
        async_add_entities(
            BackendSensor(runtime.metrics, entry, model_id, desc)
            for model_id in new
            for desc in BACKEND_SENSORS
        )

    _add_new()
    entry.async_on_unload(runtime.health.async_add_listener(_add_new))


class BackendSensor(BackendEntity[MetricsCoordinator], SensorEntity):
    """A per-backend statistic."""

    entity_description: BackendSensorDescription

    def __init__(
        self,
        coordinator: MetricsCoordinator,
        entry: LiteLLMConfigEntry,
        model_id: str,
        description: BackendSensorDescription,
    ) -> None:
        """Initialise the sensor."""
        super().__init__(coordinator, entry, model_id, description.key)
        self.entity_description = description

    async def async_added_to_hass(self) -> None:
        """Also follow backend membership from the health coordinator."""
        await super().async_added_to_hass()
        self.async_on_remove(
            self._entry.runtime_data.health.async_add_listener(self.async_write_ha_state)
        )

    @property
    def available(self) -> bool:
        """Unavailable while the proxy is down or the backend is gone."""
        data = self._entry.runtime_data.health.data
        return super().available and data.reachable and self.model_id in data.backends

    @property
    def native_value(self) -> float | None:
        """Return the value (counters read 0 until the backend serves traffic)."""
        return self.entity_description.value_fn(
            self.coordinator.data.backends.get(self.model_id, _EMPTY)
        )


class ActiveBackend(ProxyEntity[MetricsCoordinator], SensorEntity):
    """Backend that served requests in the latest poll."""

    @property
    def native_value(self) -> str:
        """Return the backend name or 'idle'."""
        return self.coordinator.data.active_backend

    @property
    def extra_state_attributes(self) -> dict[str, str | None]:
        """Expose the last backend that served anything."""
        return {"last_active_backend": self.coordinator.data.last_active_backend}


class FallbackRequests(ProxyEntity[MetricsCoordinator], SensorEntity):
    """Requests answered by a fallback model group."""

    _attr_state_class = SensorStateClass.TOTAL_INCREASING

    @property
    def native_value(self) -> int:
        """Return the total."""
        return int(self.coordinator.data.fallback_requests)


class RequestsByClient(ProxyEntity[MetricsCoordinator], SensorEntity):
    """Per-client request counts as attributes."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> int:
        """Return the number of distinct clients."""
        return len(self.coordinator.data.clients)

    @property
    def extra_state_attributes(self) -> dict[str, float]:
        """Return client_ip -> request count."""
        return dict(self.coordinator.data.clients)
