"""Constants for the LiteLLM Router integration."""

from __future__ import annotations

from datetime import timedelta
from typing import Final

DOMAIN: Final = "litellm_router"

CONF_URL: Final = "url"
CONF_API_KEY: Final = "api_key"
CONF_HEALTH_INTERVAL: Final = "health_interval"
CONF_METRICS_INTERVAL: Final = "metrics_interval"

DEFAULT_HEALTH_INTERVAL: Final = 60
DEFAULT_METRICS_INTERVAL: Final = 30
MIN_INTERVAL: Final = 10
MAX_INTERVAL: Final = 3600

# /health makes a live call to every backend and can be slow.
HEALTH_TIMEOUT: Final = 50
REQUEST_TIMEOUT: Final = 15

IDLE: Final = "idle"


def health_interval(options: dict[str, int]) -> timedelta:
    """Return the configured health poll interval."""
    return timedelta(seconds=options.get(CONF_HEALTH_INTERVAL, DEFAULT_HEALTH_INTERVAL))


def metrics_interval(options: dict[str, int]) -> timedelta:
    """Return the configured metrics poll interval."""
    return timedelta(seconds=options.get(CONF_METRICS_INTERVAL, DEFAULT_METRICS_INTERVAL))
