"""Data update coordinators."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .api import (
    Health,
    LiteLLMAuthError,
    LiteLLMClient,
    LiteLLMConnectionError,
    ModelInfo,
    Topology,
)
from .const import DOMAIN, health_interval, metrics_interval
from .prometheus import parse_prometheus
from .stats import MetricsData, MetricsProcessor

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class HealthData:
    """Result of a health poll."""

    reachable: bool = False
    health: Health = field(default_factory=Health)
    topology: Topology = field(default_factory=Topology)

    @property
    def backends(self) -> dict[str, ModelInfo]:
        """Return every known backend (configured or reported by /health)."""
        merged = dict(self.topology.models)
        for source in (self.health.unhealthy, self.health.healthy):
            for model_id, info in source.items():
                if model_id not in merged:
                    merged[model_id] = info
        return merged


class HealthCoordinator(DataUpdateCoordinator[HealthData]):
    """Polls liveliness, /health, /model/info and the configured fallbacks."""

    config_entry: ConfigEntry

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, client: LiteLLMClient) -> None:
        """Initialise the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} health",
            update_interval=health_interval(entry.options),
        )
        self.client = client

    async def _async_update_data(self) -> HealthData:
        try:
            await self.client.liveliness()
        except LiteLLMAuthError as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except LiteLLMConnectionError:
            # Keep the entry loaded: "proxy reachable" must be able to turn off.
            return HealthData(reachable=False, topology=self._previous_topology())
        try:
            health, models, fallbacks = await asyncio.gather(
                self.client.health(), self.client.model_info(), self.client.fallbacks()
            )
        except LiteLLMAuthError as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except LiteLLMConnectionError as err:
            raise UpdateFailed(f"Error talking to LiteLLM: {err}") from err
        return HealthData(True, health, Topology(models, fallbacks))

    def _previous_topology(self) -> Topology:
        return self.data.topology if self.data else Topology()


class MetricsCoordinator(DataUpdateCoordinator[MetricsData]):
    """Polls /metrics/."""

    config_entry: ConfigEntry

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        client: LiteLLMClient,
        health: HealthCoordinator,
    ) -> None:
        """Initialise the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN} metrics",
            update_interval=metrics_interval(entry.options),
        )
        self.client = client
        self.health = health
        self.processor = MetricsProcessor()

    async def _async_update_data(self) -> MetricsData:
        try:
            text = await self.client.metrics()
        except LiteLLMAuthError as err:
            raise ConfigEntryAuthFailed(str(err)) from err
        except LiteLLMConnectionError as err:
            raise UpdateFailed(f"Error fetching metrics: {err}") from err
        topology = self.health.data.topology if self.health.data else Topology()
        return self.processor.process(parse_prometheus(text), topology)
