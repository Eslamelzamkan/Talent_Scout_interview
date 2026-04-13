from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast

import httpx
import structlog
from langchain_openai import ChatOpenAI

from app.core.config import get_settings

log = structlog.get_logger()
settings = get_settings()
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 422}
P = ParamSpec("P")
T = TypeVar("T")
Provider = Literal["gemini", "openrouter"]


def _status_code_from_exception(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _usage_from_payload(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = payload.get("usage") or {}
    if not usage:
        return None, None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens", 0)
    return prompt_tokens, completion_tokens


def _with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            model = kwargs.get("model")
            if model is None and len(args) > 1:
                model = args[1]
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    status_code = _status_code_from_exception(exc)
                    if status_code in NON_RETRYABLE_STATUS_CODES:
                        log.error(
                            "llm_call_failed",
                            model=model,
                            attempt=attempt,
                            status_code=status_code,
                        )
                        raise
                    retryable = isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))
                    retryable = retryable or status_code in RETRYABLE_STATUS_CODES
                    if not retryable or attempt >= max_attempts:
                        log.error(
                            "llm_call_failed",
                            model=model,
                            attempt=attempt,
                            status_code=status_code,
                        )
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay *= random.uniform(0.5, 1.5)
                    log.warning(
                        "llm_retry",
                        model=model,
                        attempt=attempt,
                        status_code=status_code,
                        delay_seconds=round(delay, 3),
                    )
                    await asyncio.sleep(delay)
            raise RuntimeError("retry loop exited unexpectedly")

        return wrapper

    return decorator


class LLMClient:
    def __init__(
        self,
        config: Any | None = None,
        redis_client: Any | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = config or settings
        self.redis = redis_client
        self._http_override = http_client
        self._http_clients: dict[Provider, httpx.AsyncClient] = {}
        if http_client is not None:
            self._http_clients["openrouter"] = http_client
            self._http_clients["gemini"] = http_client

    def _provider_for_model(self, model: str) -> Provider:
        if "/" in model:
            return "openrouter"
        if model.startswith(("gemini-", "gemma-", "embedding-")):
            return "gemini"
        return "openrouter"

    def _client_for_provider(self, provider: Provider) -> httpx.AsyncClient:
        client = self._http_clients.get(provider)
        if client is not None:
            return client
        base_url = self._base_url_for_provider(provider)
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(30.0, read=120.0),
        )
        self._http_clients[provider] = client
        return client

    def _base_url_for_provider(self, provider: Provider) -> str:
        if provider == "gemini":
            return self.settings.gemini_base_url
        return self.settings.openrouter_base_url

    def _headers_for_provider(self, provider: Provider) -> dict[str, str]:
        if provider == "gemini":
            return self.settings.gemini_headers
        return self.settings.openrouter_headers

    def _redis_client(self) -> Any | None:
        if self.redis is not None:
            return self.redis
        import app.core as core

        return core.redis

    def _rpm_soft_limit(self, model: str) -> int:
        if model.startswith(("anthropic/", "claude-")):
            return max(int(self.settings.rpm_limit_claude), 1)
        if model.startswith(("google/gemini", "gemini-", "gemma-")):
            return max(int(self.settings.rpm_limit_gemini), 1)
        if model.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
            return max(int(self.settings.rpm_limit_gpt4), 1)
        return 200

    async def _apply_rate_limit(self, model: str) -> None:
        redis_client = self._redis_client()
        if redis_client is None:
            return
        key = f"rate:{model}:rpm"
        current = int(await redis_client.incr(key))
        await redis_client.expire(key, 60, nx=True)
        limit = self._rpm_soft_limit(model)
        if current > limit * 0.85:
            adaptive_delay = (current / limit) * 2.0
            log.warning(
                "llm_rate_limit_delay",
                model=model,
                current_rpm=current,
                adaptive_delay=round(adaptive_delay, 3),
            )
            await asyncio.sleep(adaptive_delay)

    @_with_retry(max_attempts=3, base_delay=1.0, max_delay=60.0)
    async def _post_json(
        self,
        model: str,
        path: str,
        payload: dict[str, Any],
        *,
        provider: Provider,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self._client_for_provider(provider).post(
            path,
            json=payload,
            headers=self._headers_for_provider(provider),
        )
        response.raise_for_status()
        data = cast(dict[str, Any], response.json())
        prompt_tokens, completion_tokens = _usage_from_payload(data)
        log.info(
            "llm_call",
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        return data

    @_with_retry(max_attempts=3, base_delay=1.0, max_delay=60.0)
    async def _open_stream(
        self,
        model: str,
        payload: dict[str, Any],
        *,
        provider: Provider,
    ) -> tuple[Any, httpx.Response, float]:
        started = time.perf_counter()
        stream_cm = self._client_for_provider(provider).stream(
            "POST",
            "/chat/completions",
            json=payload,
            headers=self._headers_for_provider(provider),
        )
        response = await stream_cm.__aenter__()
        try:
            response.raise_for_status()
        except Exception as exc:
            await stream_cm.__aexit__(type(exc), exc, exc.__traceback__)
            raise
        return stream_cm, response, started

    async def _stream_chat(
        self,
        model: str,
        payload: dict[str, Any],
        *,
        provider: Provider,
    ) -> AsyncGenerator[str, None]:
        stream_payload = {
            **payload,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        stream_cm, response, started = await self._open_stream(
            model=model,
            payload=stream_payload,
            provider=provider,
        )

        async def iterator() -> AsyncGenerator[str, None]:
            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            try:
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    prompt, completion = _usage_from_payload(chunk)
                    if prompt is not None:
                        prompt_tokens = prompt
                    if completion is not None:
                        completion_tokens = completion
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            yield content
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("text"):
                                    yield str(item["text"])
            finally:
                await stream_cm.__aexit__(None, None, None)
                log.info(
                    "llm_call",
                    model=model,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=round((time.perf_counter() - started) * 1000),
                )

        return iterator()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        await self._apply_rate_limit(model)
        provider = self._provider_for_model(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if stream:
            return await self._stream_chat(model=model, payload=payload, provider=provider)
        return await self._post_json(
            model=model,
            path="/chat/completions",
            payload=payload,
            provider=provider,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), 100):
            provider = self._provider_for_model(self.settings.model_embedding)
            payload = {
                "model": self.settings.model_embedding,
                "input": texts[start : start + 100],
            }
            response = await self._post_json(
                model=self.settings.model_embedding,
                path="/embeddings",
                payload=payload,
                provider=provider,
            )
            embeddings.extend(item["embedding"] for item in response.get("data", []))
        return embeddings

    def as_langchain(self, model: str, temperature: float = 0.1) -> ChatOpenAI:
        provider = self._provider_for_model(model)
        api_key = (
            self.settings.gemini_api_key
            if provider == "gemini"
            else self.settings.openrouter_api_key
        )
        headers = self._headers_for_provider(provider)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=self._base_url_for_provider(provider),
            openai_api_key=api_key,
            default_headers=headers,
        )

    async def health_check(self) -> bool:
        try:
            await self.chat(
                self.settings.model_fast,
                [{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    async def aclose(self) -> None:
        closed: set[int] = set()
        for client in self._http_clients.values():
            if id(client) in closed:
                continue
            await client.aclose()
            closed.add(id(client))

    async def close(self) -> None:
        await self.aclose()

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        stream = await self.chat(model, messages, stream=True, **kwargs)
        return cast(AsyncGenerator[str, None], stream)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]

    def langchain_chat(self, model: str) -> ChatOpenAI:
        return self.as_langchain(model)
