"""API client."""

import aiohttp
import pytest

from custom_components.litellm_router.api import (
    LiteLLMAuthError,
    LiteLLMClient,
    LiteLLMConnectionError,
    normalize_url,
    parse_fallbacks,
    parse_health,
    parse_model_info,
)

from .conftest import API_KEY, HEALTH, LOCAL, MODEL_INFO, URL


def test_normalize_url() -> None:
    assert normalize_url(" host:4000/ ") == "http://host:4000"
    assert normalize_url("https://h/") == "https://h"


def test_parsers_tolerate_junk() -> None:
    assert (
        parse_health({"healthy_endpoints": [{"model": "x"}], "unhealthy_endpoints": None}).healthy
        == {}
    )
    assert parse_health(HEALTH).unhealthy
    assert parse_model_info({"data": [{"model_name": "n", "model_info": {}}]}) == {}
    assert parse_model_info(MODEL_INFO)[LOCAL].name == "qwen-local"
    assert parse_fallbacks(
        {"current_values": {"fallbacks": ["x", {"a": ["b"]}, {"c": "d"}, {"a": ["e"]}]}}
    ) == {"a": ["b", "e"]}
    assert parse_fallbacks({}) == {}


async def test_requests(hass, aioclient_mock) -> None:
    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    c = LiteLLMClient(async_get_clientsession(hass), URL + "/", API_KEY)
    aioclient_mock.get(f"{URL}/health/liveliness", text="ok")
    aioclient_mock.get(f"{URL}/metrics/", text="m 1")
    aioclient_mock.get(f"{URL}/health", json=HEALTH)
    aioclient_mock.get(f"{URL}/model/info", json=MODEL_INFO)
    aioclient_mock.get(
        f"{URL}/router/settings", json={"current_values": {"fallbacks": [{"a": ["b"]}]}}
    )
    await c.liveliness()
    assert await c.metrics() == "m 1"
    assert len((await c.health()).healthy) == 2
    assert LOCAL in await c.model_info()
    assert await c.fallbacks() == {"a": ["b"]}
    assert aioclient_mock.mock_calls[0][3]["Authorization"] == f"Bearer {API_KEY}"


@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"status": 401}, LiteLLMAuthError),
        ({"status": 403}, LiteLLMAuthError),
        ({"status": 500}, LiteLLMConnectionError),
        ({"exc": aiohttp.ClientError("boom")}, LiteLLMConnectionError),
        ({"exc": TimeoutError()}, LiteLLMConnectionError),
    ],
)
async def test_errors(hass, aioclient_mock, kwargs, exc) -> None:
    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    c = LiteLLMClient(async_get_clientsession(hass), URL, API_KEY)
    aioclient_mock.get(f"{URL}/health/liveliness", **kwargs)
    with pytest.raises(exc):
        await c.liveliness()


async def test_bad_json_and_fallbacks_degrade(hass, aioclient_mock) -> None:
    from homeassistant.helpers.aiohttp_client import async_get_clientsession

    c = LiteLLMClient(async_get_clientsession(hass), URL, API_KEY)
    aioclient_mock.get(f"{URL}/health", text="not json")
    aioclient_mock.get(f"{URL}/model/info", json=[1])
    aioclient_mock.get(f"{URL}/router/settings", status=500)
    with pytest.raises(LiteLLMConnectionError):
        await c.health()
    with pytest.raises(LiteLLMConnectionError):
        await c.model_info()
    assert await c.fallbacks() == {}
