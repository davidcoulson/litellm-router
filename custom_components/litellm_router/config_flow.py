"""Config, reauth and options flows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import LiteLLMAuthError, LiteLLMClient, LiteLLMConnectionError, normalize_url
from .const import (
    CONF_API_KEY,
    CONF_HEALTH_INTERVAL,
    CONF_METRICS_INTERVAL,
    CONF_URL,
    DEFAULT_HEALTH_INTERVAL,
    DEFAULT_METRICS_INTERVAL,
    DOMAIN,
    MAX_INTERVAL,
    MIN_INTERVAL,
)

_INTERVAL = vol.All(vol.Coerce(int), vol.Range(min=MIN_INTERVAL, max=MAX_INTERVAL))


async def _validate(hass: Any, url: str, api_key: str) -> str | None:
    """Return an error key, or None when the proxy and key are good."""
    client = LiteLLMClient(async_get_clientsession(hass), url, api_key)
    try:
        await client.liveliness()
        await client.metrics()
    except LiteLLMAuthError:
        return "invalid_auth"
    except LiteLLMConnectionError:
        return "cannot_connect"
    return None


class LiteLLMRouterConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the config flow."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Ask for the proxy URL and API key."""
        errors: dict[str, str] = {}
        if user_input is not None:
            url = normalize_url(user_input[CONF_URL])
            await self.async_set_unique_id(url, raise_on_progress=False)
            self._abort_if_unique_id_configured()
            if error := await _validate(self.hass, url, user_input[CONF_API_KEY]):
                errors["base"] = error
            else:
                # Re-check after the awaited validation: another flow may have won.
                await self.async_set_unique_id(url, raise_on_progress=False)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title=f"LiteLLM ({url})",
                    data={CONF_URL: url, CONF_API_KEY: user_input[CONF_API_KEY]},
                )
        schema = vol.Schema(
            {
                vol.Required(CONF_URL, default=(user_input or {}).get(CONF_URL, "http://")): str,
                vol.Required(CONF_API_KEY): str,
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async def async_step_reauth(self, entry_data: Mapping[str, Any]) -> ConfigFlowResult:
        """Start reauthentication."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for a new API key."""
        entry = self._get_reauth_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            if error := await _validate(self.hass, entry.data[CONF_URL], user_input[CONF_API_KEY]):
                errors["base"] = error
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates={CONF_API_KEY: user_input[CONF_API_KEY]}
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema({vol.Required(CONF_API_KEY): str}),
            description_placeholders={"url": entry.data[CONF_URL]},
            errors=errors,
        )

    def _url_taken(self, entry: ConfigEntry, url: str) -> bool:
        return any(
            other.unique_id == url
            for other in self._async_current_entries(include_ignore=False)
            if other.entry_id != entry.entry_id
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Change the proxy URL and, optionally, the API key."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            url = normalize_url(user_input[CONF_URL])
            api_key = user_input.get(CONF_API_KEY) or entry.data[CONF_API_KEY]
            if self._url_taken(entry, url):
                return self.async_abort(reason="already_configured")
            if error := await _validate(self.hass, url, api_key):
                errors["base"] = error
            else:
                # Re-check after the awaited validation (no await before the update).
                if self._url_taken(entry, url):
                    return self.async_abort(reason="already_configured")
                return self.async_update_reload_and_abort(
                    entry,
                    unique_id=url,
                    data_updates={CONF_URL: url, CONF_API_KEY: api_key},
                )
        schema = vol.Schema(
            {
                vol.Required(CONF_URL, default=entry.data[CONF_URL]): str,
                vol.Optional(CONF_API_KEY): str,
            }
        )
        return self.async_show_form(step_id="reconfigure", data_schema=schema, errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow."""
        return LiteLLMRouterOptionsFlow()


class LiteLLMRouterOptionsFlow(OptionsFlow):
    """Poll interval options."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Edit the poll intervals."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        opts = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_HEALTH_INTERVAL,
                    default=opts.get(CONF_HEALTH_INTERVAL, DEFAULT_HEALTH_INTERVAL),
                ): _INTERVAL,
                vol.Required(
                    CONF_METRICS_INTERVAL,
                    default=opts.get(CONF_METRICS_INTERVAL, DEFAULT_METRICS_INTERVAL),
                ): _INTERVAL,
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
