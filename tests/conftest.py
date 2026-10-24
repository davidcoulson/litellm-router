"""Shared fixtures. The API key here is fake."""

from __future__ import annotations

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.litellm_router.const import CONF_API_KEY, CONF_URL, DOMAIN

URL = "http://litellm.test:4000"
API_KEY = "sk-fake-test-key"
LOCAL = "a" * 64
FALLBACK = "b" * 64
FAST = "c" * 64

HEALTH = {
    "healthy_endpoints": [
        {"api_base": "http://strata/v1", "model": "openai/qwen", "model_id": LOCAL},
        {"api_base": "http://ollama", "model": "ollama_chat/qwen3:8b", "model_id": FALLBACK},
    ],
    "unhealthy_endpoints": [
        {"api_base": "http://ollama", "model": "ollama_chat/qwen3:8b", "model_id": FAST}
    ],
    "healthy_count": 2,
    "unhealthy_count": 1,
}
MODEL_INFO = {
    "data": [
        {
            "model_name": "qwen-local",
            "litellm_params": {"api_base": "http://strata/v1", "model": "openai/qwen"},
            "model_info": {"id": LOCAL},
        },
        {
            "model_name": "qwen-fallback",
            "litellm_params": {"api_base": "http://ollama", "model": "ollama_chat/qwen3:8b"},
            "model_info": {"id": FALLBACK},
        },
        {
            "model_name": "qwen-fast",
            "litellm_params": {"api_base": "http://ollama", "model": "ollama_chat/qwen3:8b"},
            "model_info": {"id": FAST},
        },
    ]
}
ROUTER_SETTINGS = {"current_values": {"fallbacks": [{"qwen-local": ["qwen-fallback"]}]}}


def req(model_id, requested, value, client="10.0.0.9", status="200"):
    """Return a request-counter line."""
    return (
        f'litellm_proxy_total_requests_metric_total{{client_ip="{client}",model_id="{model_id}",'
        f'requested_model="{requested}",route="/v1/chat/completions",status_code="{status}",'
        f'user_agent="curl, \\"x\\" {{1}}"}} {value}'
    )


def metric(name, model_id, value, **extra):
    """Return a simple labelled metric line."""
    labels = ",".join(f'{k}="{v}"' for k, v in {"model_id": model_id, **extra}.items())
    return f"{name}{{{labels}}} {value}"


def metrics_text(*lines: str) -> str:
    """Join lines into exposition text."""
    return "# HELP x y\n# TYPE x counter\n" + "\n".join(lines) + "\n"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations."""


@pytest.fixture
def entry() -> MockConfigEntry:
    """Return a config entry."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="LiteLLM",
        unique_id=URL,
        data={CONF_URL: URL, CONF_API_KEY: API_KEY},
    )


def mock_proxy(aioclient_mock, metrics="", health=HEALTH, metrics_status=None, **overrides):
    """Register the proxy endpoints; overrides maps a path key to aiohttp mock kwargs."""
    aioclient_mock.clear_requests()
    routes = {
        "liveliness": ("/health/liveliness", {"text": "I'm alive!"}),
        "health_ep": ("/health", {"json": health}),
        "model_info": ("/model/info", {"json": MODEL_INFO}),
        "router": ("/router/settings", {"json": ROUTER_SETTINGS}),
        "metrics_ep": (
            "/metrics/",
            {"status": metrics_status} if metrics_status else {"text": metrics},
        ),
    }
    for key, (path, kwargs) in routes.items():
        aioclient_mock.get(f"{URL}{path}", **overrides.get(key, kwargs))
