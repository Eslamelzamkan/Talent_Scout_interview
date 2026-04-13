from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.core.llm import LLMClient

llm_module = importlib.import_module("app.core.llm")


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        openrouter_api_key="test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        gemini_api_key="gemini-key",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        openrouter_headers={
            "Authorization": "Bearer test-key",
            "HTTP-Referer": "https://talentscout.ai",
            "X-Title": "Talent Scout",
        },
        gemini_headers={"Authorization": "Bearer gemini-key"},
        model_fast="google/gemini-2.0-flash-001",
        model_embedding="openai/text-embedding-3-large",
        rpm_limit_gpt4=500,
        rpm_limit_claude=400,
        rpm_limit_gemini=1000,
    )


def _response(status_code: int, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
        json=payload,
    )


@pytest.mark.asyncio
async def test_retry_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(llm_module.asyncio, "sleep", sleep)
    monkeypatch.setattr(llm_module.random, "uniform", lambda _a, _b: 1.0)
    http_client = MagicMock()
    http_client.post = AsyncMock(
        side_effect=[
            _response(429, {"error": "slow down"}),
            _response(429, {"error": "slow down"}),
            _response(
                200,
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            ),
        ]
    )
    http_client.aclose = AsyncMock()
    client = LLMClient(_settings(), http_client=http_client)

    result = await client.chat("openai/gpt-4.1", [{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "ok"
    assert http_client.post.await_count == 3
    assert sleep.await_count == 2


@pytest.mark.asyncio
async def test_no_retry_on_400(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(llm_module.asyncio, "sleep", sleep)
    http_client = MagicMock()
    http_client.post = AsyncMock(side_effect=[_response(400, {"error": "bad request"})])
    http_client.aclose = AsyncMock()
    client = LLMClient(_settings(), http_client=http_client)

    with pytest.raises(httpx.HTTPStatusError):
        await client.chat("openai/gpt-4.1", [{"role": "user", "content": "hi"}])

    assert http_client.post.await_count == 1
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_embed_batching() -> None:
    batch_sizes: list[int] = []

    async def post(
        _path: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        batch = list(json["input"])
        batch_sizes.append(len(batch))
        payload = {
            "data": [{"embedding": [float(index)]} for index, _ in enumerate(batch)],
            "usage": {"prompt_tokens": len(batch)},
        }
        return _response(200, payload)

    http_client = MagicMock()
    http_client.post = AsyncMock(side_effect=post)
    http_client.aclose = AsyncMock()
    client = LLMClient(_settings(), http_client=http_client)

    embeddings = await client.embed([f"text-{index}" for index in range(150)])

    assert batch_sizes == [100, 50]
    assert len(embeddings) == 150
    assert http_client.post.await_count == 2


@pytest.mark.asyncio
async def test_gemini_models_use_gemini_headers() -> None:
    captured_headers: list[dict[str, str]] = []

    async def post(
        _path: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        captured_headers.append(headers)
        return _response(
            200,
            {
                "choices": [{"message": {"content": f"ok:{json['model']}"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    http_client = MagicMock()
    http_client.post = AsyncMock(side_effect=post)
    http_client.aclose = AsyncMock()
    client = LLMClient(_settings(), http_client=http_client)

    result = await client.chat("gemini-2.0-flash", [{"role": "user", "content": "hi"}])

    assert result["choices"][0]["message"]["content"] == "ok:gemini-2.0-flash"
    assert captured_headers == [{"Authorization": "Bearer gemini-key"}]


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):  # type: ignore[no-untyped-def]
        for line in self._lines:
            yield line


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self.response = response
        self.exited = False

    async def __aenter__(self) -> _FakeStreamResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.exited = True


@pytest.mark.asyncio
async def test_streaming_assembly() -> None:
    stream_context = _FakeStreamContext(
        _FakeStreamResponse(
            [
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                (
                    'data: {"usage":{"prompt_tokens":3,"completion_tokens":2},'
                    '"choices":[{"delta":{}}]}'
                ),
                "data: [DONE]",
            ]
        )
    )
    http_client = MagicMock()
    http_client.stream = MagicMock(return_value=stream_context)
    http_client.aclose = AsyncMock()
    client = LLMClient(_settings(), http_client=http_client)

    stream = await client.chat(
        "openai/gpt-4.1",
        [{"role": "user", "content": "hi"}],
        stream=True,
    )
    assembled = "".join([chunk async for chunk in stream])

    assert assembled == "Hello world"
    assert stream_context.exited is True


@pytest.mark.asyncio
async def test_rate_limiter_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(llm_module.asyncio, "sleep", sleep)
    redis_client = MagicMock()
    redis_client.incr = AsyncMock(return_value=450)
    redis_client.expire = AsyncMock()
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(
            200,
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )
    )
    http_client.aclose = AsyncMock()
    client = LLMClient(_settings(), redis_client=redis_client, http_client=http_client)

    await client.chat("openai/gpt-4.1", [{"role": "user", "content": "hi"}])

    assert sleep.await_count == 1
    assert sleep.await_args.args[0] > 0


@pytest.mark.asyncio
async def test_rate_limiter_skips_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr(llm_module.asyncio, "sleep", sleep)
    redis_client = MagicMock()
    redis_client.incr = AsyncMock(return_value=10)
    redis_client.expire = AsyncMock()
    http_client = MagicMock()
    http_client.post = AsyncMock(
        return_value=_response(
            200,
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )
    )
    http_client.aclose = AsyncMock()
    client = LLMClient(_settings(), redis_client=redis_client, http_client=http_client)

    await client.chat("openai/gpt-4.1", [{"role": "user", "content": "hi"}])

    sleep.assert_not_awaited()
