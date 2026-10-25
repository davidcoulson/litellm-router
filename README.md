# LiteLLM Router for Home Assistant

Monitor what a [LiteLLM](https://github.com/BerriAI/litellm) proxy routes where: backend health, requests, failures,
tokens, latency and fallbacks. Local polling, configured entirely in the UI, no YAML and no REST sensors, and no
database needed on the proxy (unlike the spend API).

## Requirements

* LiteLLM proxy with the Prometheus callback enabled (`litellm_settings: callbacks: ["prometheus"]`), so that
  `/metrics/` exists.
* An API key that may read `/health`, `/model/info`, `/router/settings` and `/metrics/`.
* Home Assistant 2025.1 or newer.

## Install

HACS: add this repository as a custom repository (category Integration), install **LiteLLM Router**, restart
Home Assistant. Manual: copy `custom_components/litellm_router` into your `config/custom_components/` and restart.

Then Settings > Devices & services > Add integration > **LiteLLM Router**, and enter the proxy URL
(for example `http://10.2.6.57:4000`) and the API key. The key is validated against `/health/liveliness` and
`/metrics/`. If the key is later rejected, Home Assistant starts a re-authentication flow.

To change the URL or key later, use Reconfigure on the integration (leave the key empty to keep it).

Options (Configure on the integration): health poll interval (default 60 s) and metrics poll interval (default 30 s).
`/health` makes a live call to each backend, so it is polled separately and less often.

## Entities

One device for the proxy, plus one device per backend (keyed by `model_id`; new backends are added automatically,
and a backend that disappears goes unavailable).

Proxy device:

| Entity | Type | Notes |
| --- | --- | --- |
| Proxy reachable | binary_sensor (connectivity) | `/health/liveliness` answers |
| Active backend | sensor | Backend that served the most requests in the latest metrics poll, or `idle`. Attribute `last_active_backend` |
| Fallback requests | sensor, total_increasing | Requests served by a model group that is a fallback of another group, where the requested model differs from the one that answered |
| Requests by client | sensor, diagnostic | State is the number of clients; each `client_ip` is an attribute with its request count |

Per backend:

| Entity | Type | Notes |
| --- | --- | --- |
| Healthy | binary_sensor (connectivity) | Healthy per `/health`. Attributes `api_base`, `model` |
| Requests served | sensor, total_increasing | Successful (2xx) requests |
| Failures | sensor, total_increasing | From `litellm_proxy_failed_requests_metric_total` |
| Tokens in / Tokens out | sensors, total_increasing | |
| Average API latency | sensor, measurement, s | Delta of histogram sum/count over the last poll |
| Average time to first token | sensor, measurement, s | Same; only populated for streaming requests |

Behaviour notes:

* Totals are accumulated from per-poll deltas and never go negative. When the LiteLLM container restarts (counters
  drop) the new value is counted as the delta, so the sensors keep rising. Totals are saved per config entry
  (Home Assistant storage, deleted with the entry) and restored on reload and restart, so they continue where they
  were; only a first-ever setup starts from the proxy's current counters.
* Per-source counter state is bounded: sources unseen for 24 h with nothing pending are rolled into the totals and
  dropped (at most 5000 are kept). A dropped source that returns continues from its last reading. More than 5000
  distinct label sets is unsupported: once the cap is reached new series are ignored (one warning is logged), and
  per-client counts beyond 500 clients are grouped as `other`.
* Average latency and time to first token keep their last value during idle polls (an interval with no new requests
  has no average).
* Series without a `model_id` (the proxy's own health probes) are ignored.
* The "fallback" test uses `/router/settings` fallbacks plus the `requested_model` label. If the fallback config
  can't be read, any request answered by a different group than requested counts.
* Failures come from the proxy-level failure counter; its labels vary between LiteLLM versions. A failure sample
  without `model_id` is attributed to the requested model group only if that group has exactly one deployment.

## Diagnostics

Download diagnostics from the device page; the API key is redacted and credentials in URLs (userinfo and
credential-like query values, including backend `api_base`) are stripped everywhere they are shown.

## Development

```
uv venv ~/venvs/litellm-router --python 3.13
uv pip install --python ~/venvs/litellm-router/bin/python -r requirements_test.txt
~/venvs/litellm-router/bin/python -m pytest
```
