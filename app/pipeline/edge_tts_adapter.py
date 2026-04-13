from __future__ import annotations

import importlib
import uuid

from livekit.agents import APIConnectOptions
from livekit.agents.tts import TTS, AudioEmitter, ChunkedStream, TTSCapabilities


class EdgeTTS(TTS):
    def __init__(self, *, voice: str) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._voice = voice

    @property
    def model(self) -> str:
        return self._voice

    @property
    def provider(self) -> str:
        return "edge-tts"

    def synthesize(  # type: ignore[override]
        self,
        text: str,
        *,
        conn_options: APIConnectOptions,
    ) -> ChunkedStream:
        return _EdgeTTSChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    async def aclose(self) -> None:
        return None


class _EdgeTTSChunkedStream(ChunkedStream):
    async def _run(self, output_emitter: AudioEmitter) -> None:
        edge_tts = importlib.import_module("edge_tts")
        communicate = edge_tts.Communicate(text=self.input_text, voice=self._tts.model)

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/mpeg",
            stream=False,
        )

        async for chunk in communicate.stream():
            if chunk["type"] != "audio":
                continue
            output_emitter.push(chunk["data"])

        output_emitter.flush()
