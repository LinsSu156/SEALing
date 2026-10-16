"""User-configured provider transport and response parsing."""

from __future__ import annotations

import http.client
import json
import os
import time
from dataclasses import dataclass
from typing import Tuple

from .profiles import API_SYSTEM_PROMPT


@dataclass(frozen=True)
class LLMProviderConfig:
    display_name: str
    provider: str
    api_model_id: str
    base_url: str


class LLMQuotaExhaustedError(RuntimeError):
    """Raised when the upstream LLM provider reports quota/resource exhaustion."""


class LLMProviderResponseError(RuntimeError):
    """Raised when the upstream LLM provider returns an empty or unparsable response."""


PROVIDERS = {
    "openai-compatible": ("https://api.openai.com/v1", "OPENAI"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "GEMINI"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK"),
}


def get_llm_provider_config(
    model_name: str,
    api_model_id: str = None,
    provider: str = None,
    base_url: str = None,
) -> LLMProviderConfig:
    """Resolve explicit settings, then environment settings, then public defaults."""
    name = str(model_name)
    provider = provider or os.environ.get("LLM_PROVIDER")
    if provider is None:
        provider = next(
            (
                p
                for token, p in (
                    ("claude", "anthropic"),
                    ("gemini", "gemini"),
                    ("deepseek", "deepseek"),
                )
                if token in name.lower()
            ),
            "openai-compatible",
        )
    provider = {"openai": "openai-compatible"}.get(provider, provider)
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r}; choose from {', '.join(PROVIDERS)}"
        )
    default_url, env_prefix = PROVIDERS[provider]
    url = (base_url or os.environ.get(f"{env_prefix}_BASE_URL") or default_url).rstrip(
        "/"
    )
    return LLMProviderConfig(
        name, provider, api_model_id or os.environ.get("LLM_API_MODEL_ID") or name, url
    )


def _resolve_provider_api_settings(
    provider_config: LLMProviderConfig, api_key=None, base_url=None
) -> Tuple[str | None, str]:
    _, prefix = PROVIDERS[provider_config.provider]
    return api_key or os.environ.get(f"{prefix}_API_KEY"), (
        base_url or provider_config.base_url
    ).rstrip("/")


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if not isinstance(obj, dict):
            raise ValueError("Expected a JSON object for semantic evidence")
        return obj
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError(
        f"No JSON object found in API response. Response prefix: {text[:500]!r}"
    )


def _post_json(
    base_url: str, path: str, payload: dict, headers: dict, timeout: int
) -> Tuple[int, str]:
    """Append an API route to a user-supplied base URL, retaining its path prefix."""
    from urllib.parse import urlsplit

    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTP(S) URL without a query or fragment")
    connection = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = connection(parsed.netloc, timeout=timeout)
    try:
        route = parsed.path.rstrip("/") + "/" + path.lstrip("/")
        conn.request("POST", route, json.dumps(payload).encode("utf-8"), headers)
        response = conn.getresponse()
        return response.status, response.read().decode("utf-8")
    finally:
        conn.close()


def _extract_gemini_text(obj) -> str:
    texts = []
    if isinstance(obj, dict):
        for candidate in obj.get("candidates", []) or []:
            content = (
                candidate.get("content", {}) if isinstance(candidate, dict) else {}
            )
            for part in content.get("parts", []) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        for key in ("text", "output_text"):
            if isinstance(obj.get(key), str):
                texts.append(obj[key])
    return "".join(texts)


def _extract_openai_chat_text(obj) -> str:
    texts = []
    if isinstance(obj, dict):
        for choice in obj.get("choices", []) or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                texts.append(message["content"])
            delta = choice.get("delta", {})
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                texts.append(delta["content"])
    return "".join(texts)


def _raise_if_provider_error(data: str, provider_name: str) -> None:
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return
    error = obj.get("error") if isinstance(obj, dict) else None
    if error:
        message = str(error.get("message", error) if isinstance(error, dict) else error)
        if any(
            token in message.lower()
            for token in ("quota", "rate limit", "resource has been exhausted")
        ):
            raise LLMQuotaExhaustedError(f"{provider_name}: {message}")
        raise RuntimeError(f"{provider_name} API error: {message}")


def _call_llm_json_api(
    user_prompt: str,
    api_key: str,
    base_url: str,
    model_name: str,
    api_model_id: str = None,
    provider: str = None,
    max_retries: int = 3,
    timeout: int = 120,
    system_prompt: str = API_SYSTEM_PROMPT,
) -> Tuple[dict, str]:
    config = get_llm_provider_config(model_name, api_model_id, provider, base_url)
    headers = {"Content-Type": "application/json"}
    if config.provider in {"openai-compatible", "deepseek"}:
        path = "chat/completions"
        payload = {
            "model": config.api_model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        headers["Authorization"] = f"Bearer {api_key}"
        extract_text = _extract_openai_chat_text
    elif config.provider == "anthropic":
        path = "messages"
        payload = {
            "model": config.api_model_id,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 4096,
            "stream": False,
        }
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        extract_text = lambda obj: "".join(
            block.get("text", "")
            for block in obj.get("content", [])
            if block.get("type") == "text"
        )
    else:
        from urllib.parse import quote

        path = f"models/{quote(config.api_model_id, safe='')}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        headers["x-goog-api-key"] = api_key
        extract_text = _extract_gemini_text
    if max_retries < 1:
        raise ValueError("max_retries must be positive")
    last_error = None
    for attempt in range(max_retries):
        try:
            status, data = _post_json(config.base_url, path, payload, headers, timeout)
            if status == 429:
                raise LLMQuotaExhaustedError(
                    f"{config.provider} returned HTTP 429; resume after the limit resets"
                )
            _raise_if_provider_error(data, config.provider)
            if not 200 <= status < 300:
                raise RuntimeError(f"{config.provider} API returned HTTP {status}")
            try:
                content = extract_text(json.loads(data))
                summary = _extract_json_object(content)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise LLMProviderResponseError(
                    f"{config.provider} returned invalid JSON evidence"
                ) from exc
            return summary, content
        except LLMQuotaExhaustedError:
            raise
        except (OSError, http.client.HTTPException, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(2**attempt)
    raise last_error
