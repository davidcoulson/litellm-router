"""Client for the LiteLLM proxy."""

from __future__ import annotations

import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from .const import HEALTH_TIMEOUT, REQUEST_TIMEOUT


class LiteLLMError(Exception):
    """Base error."""


class LiteLLMConnectionError(LiteLLMError):
    """The proxy could not be reached or returned a server error."""


class LiteLLMAuthError(LiteLLMError):
    """The API key was rejected."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """A deployment (backend) as configured in the proxy."""

    model_id: str
    name: str
    api_base: str = ""
    model: str = ""


@dataclass(slots=True)
class Topology:
    """Deployments and fallbacks."""

    models: dict[str, ModelInfo] = field(default_factory=dict)
    # model_group -> list of fallback model_groups
    fallbacks: dict[str, list[str]] = field(default_factory=dict)

    @property
    def fallback_targets(self) -> set[str]:
        """Return every model group that is a fallback of some other group."""
        return {t for targets in self.fallbacks.values() for t in targets}


@dataclass(slots=True)
class Health:
    """Parsed /health response."""

    healthy: dict[str, ModelInfo] = field(default_factory=dict)
    unhealthy: dict[str, ModelInfo] = field(default_factory=dict)


def normalize_url(url: str) -> str:
    """Strip whitespace and trailing slashes; default to http://."""
    url = url.strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    return url


_SECRET_PARTS = ("key", "token", "secret", "password", "passwd", "pwd", "auth", "sig", "credential")


def sanitize_url(url: str) -> str:
    """Drop userinfo and fragment and redact credential-like query values."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        netloc = host + (f":{parts.port}" if parts.port else "")
    except ValueError:
        return "**INVALID_URL**"
    query = urlencode(
        [
            (k, "**REDACTED**" if any(p in k.lower() for p in _SECRET_PARTS) else v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
        ],
        safe="*",
    )
    return urlunsplit((parts.scheme, netloc, parts.path, query, ""))


def parse_health(data: dict[str, Any]) -> Health:
    """Parse the /health payload."""
    health = Health()
    for key, target in (
        ("healthy_endpoints", health.healthy),
        ("unhealthy_endpoints", health.unhealthy),
    ):
        for item in data.get(key) or []:
            model_id = item.get("model_id")
            if not model_id:
                continue
            model = item.get("model") or ""
            target[model_id] = ModelInfo(
                model_id, model or model_id, item.get("api_base") or "", model
            )
    return health


def parse_model_info(data: dict[str, Any]) -> dict[str, ModelInfo]:
    """Parse /model/info into model_id -> ModelInfo."""
    models: dict[str, ModelInfo] = {}
    for item in data.get("data") or []:
        params = item.get("litellm_params") or {}
        model_id = (item.get("model_info") or {}).get("id")
        if not model_id:
            continue
        models[model_id] = ModelInfo(
            model_id,
            item.get("model_name") or model_id,
            params.get("api_base") or "",
            params.get("model") or "",
        )
    return models


def parse_fallbacks(data: dict[str, Any]) -> dict[str, list[str]]:
    """Parse the configured fallbacks out of /router/settings."""
    raw = (data.get("current_values") or {}).get("fallbacks") or []
    result: dict[str, list[str]] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        for group, targets in entry.items():
            if isinstance(targets, list):
                result.setdefault(group, []).extend(str(t) for t in targets)
    return result


class LiteLLMClient:
    """Thin async client."""

    def __init__(self, session: aiohttp.ClientSession, url: str, api_key: str) -> None:
        """Initialise the client."""
        self._session = session
        self._url = normalize_url(url)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def _get(self, path: str, timeout: int = REQUEST_TIMEOUT) -> tuple[int, str]:
        try:
            async with self._session.get(
                f"{self._url}{path}",
                headers=self._headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                text = await resp.text()
                status = resp.status
        except (aiohttp.ClientError, TimeoutError) as err:
            raise LiteLLMConnectionError(str(err) or type(err).__name__) from err
        if status in (401, 403):
            raise LiteLLMAuthError(f"HTTP {status}")
        if status >= 400:
            raise LiteLLMConnectionError(f"HTTP {status} from {path}")
        return status, text

    async def _get_json(self, path: str, timeout: int = REQUEST_TIMEOUT) -> dict[str, Any]:
        _, text = await self._get(path, timeout)
        try:
            data = json.loads(text)
        except ValueError as err:
            raise LiteLLMConnectionError(f"Invalid JSON from {path}") from err
        if not isinstance(data, dict):
            raise LiteLLMConnectionError(f"Unexpected JSON from {path}")
        return data

    async def liveliness(self) -> None:
        """Raise unless the proxy answers /health/liveliness."""
        await self._get("/health/liveliness")

    async def health(self) -> Health:
        """Return backend health (live calls to each backend; slow)."""
        return parse_health(await self._get_json("/health", HEALTH_TIMEOUT))

    async def metrics(self) -> str:
        """Return the Prometheus text (trailing slash avoids a 307)."""
        _, text = await self._get("/metrics/")
        return text

    async def model_info(self) -> dict[str, ModelInfo]:
        """Return configured deployments."""
        return parse_model_info(await self._get_json("/model/info"))

    async def fallbacks(self) -> dict[str, list[str]] | None:
        """Return configured fallbacks, or None when unavailable."""
        try:
            return parse_fallbacks(await self._get_json("/router/settings"))
        except LiteLLMConnectionError:
            return None
