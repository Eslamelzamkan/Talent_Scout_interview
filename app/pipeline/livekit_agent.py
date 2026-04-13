from __future__ import annotations

import asyncio
import importlib.util
from functools import lru_cache
from typing import Any

import structlog

from app.core.config import get_settings
from app.pipeline.edge_tts_adapter import EdgeTTS
from app.pipeline.interview import graph_invoke_config

try:
    from livekit.agents import Agent as VoicePipelineAgent
except Exception:  # pragma: no cover - optional runtime dependency surface

    class VoicePipelineAgent:  # type: ignore[no-redef]
        session: Any = None

        def __init__(self, *_: Any, **__: Any) -> None:
            pass


log = structlog.get_logger()
settings = get_settings()


@lru_cache(maxsize=1)
def _build_vad() -> Any | None:
    try:
        from livekit.plugins import silero
    except Exception as exc:  # pragma: no cover - optional runtime dependency surface
        log.warning("silero_vad_unavailable", error=str(exc))
        return None

    try:
        return silero.VAD.load()
    except Exception as exc:  # pragma: no cover - runtime initialization surface
        log.warning("silero_vad_init_failed", error=str(exc))
        return None


def _build_stt() -> Any | None:
    if not settings.groq_api_key:
        return None
    try:
        from livekit.plugins import groq
    except Exception as exc:  # pragma: no cover - optional runtime dependency surface
        log.warning("groq_stt_unavailable", error=str(exc))
        return None

    try:
        return groq.STT(
            model=settings.groq_stt_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            language="en",
        )
    except Exception as exc:  # pragma: no cover - runtime initialization surface
        log.warning("groq_stt_init_failed", error=str(exc))
        return None


def _build_tts() -> Any | None:
    if importlib.util.find_spec("edge_tts") is None:
        log.warning("edge_tts_unavailable")
        return None
    try:
        return EdgeTTS(voice=settings.edge_tts_voice)
    except Exception as exc:  # pragma: no cover - runtime initialization surface
        log.warning("edge_tts_init_failed", error=str(exc))
        return None


class TalentScoutAgent(VoicePipelineAgent):
    def __init__(
        self,
        session_id: str,
        candidate_id: str,
        job_id: str,
        graph: Any,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        vad = _build_vad()
        stt = _build_stt()
        tts = _build_tts()
        if stt is not None and vad is None and not getattr(stt.capabilities, "streaming", False):
            log.warning("stt_disabled_without_vad", model=settings.groq_stt_model)
            stt = None
        turn_detection = "vad" if vad is not None else "stt" if stt is not None else None
        super().__init__(
            instructions="Run the Talent Scout interview flow through LangGraph only.",
            vad=vad,
            stt=stt,
            llm=None,
            tts=tts,
            turn_detection=turn_detection,
            allow_interruptions=True,
        )
        self.session_id = session_id
        self.candidate_id = candidate_id
        self.job_id = job_id
        self.graph = graph
        self.initial_state = initial_state
        self.thread_config = graph_invoke_config(session_id, session_id)
        self.voice_pipeline = {
            "vad": "silero" if vad is not None else "disabled",
            "stt_primary": f"groq:{settings.groq_stt_model}" if stt is not None else "disabled",
            "stt_fallback": "disabled",
            "tts": f"edge-tts:{settings.edge_tts_voice}" if tts is not None else "disabled",
        }

    async def on_enter(self) -> None:
        payload = self.initial_state
        if payload is None:
            snapshot = await self.graph.aget_state(self.thread_config)
            if not snapshot.values:
                raise RuntimeError(
                    "TalentScoutAgent requires initial_state or an existing checkpoint"
                )
            state = snapshot.values
        else:
            state = await self.graph.ainvoke(payload, self.thread_config)
        question = state.get("current_question")
        if question:
            await self.say(question, allow_interruptions=True)

    async def on_user_turn_completed(self, _: Any, new_message: Any) -> None:
        transcript = (
            getattr(new_message, "text_content", None)
            or getattr(new_message, "content", None)
            or ""
        )
        if isinstance(transcript, list):
            transcript = " ".join(str(item) for item in transcript)
        await self.on_user_speech_committed(str(transcript).strip())

    async def on_user_speech_committed(self, event: Any) -> None:
        transcript = self._transcript_text(event)
        if not transcript:
            return
        await self.graph.aupdate_state(self.thread_config, {"candidate_answer": transcript})
        state = await self.graph.ainvoke(None, self.thread_config)
        if state.get("interview_complete"):
            await self.say("Thank you. That concludes our interview.", allow_interruptions=False)
            await asyncio.sleep(2)
            await self.disconnect()
            return
        question = state.get("current_question")
        if question:
            await self.say(question, allow_interruptions=True)

    async def on_stt_error(self, error: Exception) -> None:
        log.warning(
            "stt_primary_failed",
            session_id=self.session_id,
            candidate_id=self.candidate_id,
            error=str(error),
            fallback_model=self.voice_pipeline["stt_fallback"],
        )
        await self.graph.aupdate_state(self.thread_config, {"stt_fallback_active": True})

    async def say(self, text: str, *, allow_interruptions: bool) -> None:
        if not text:
            return
        session = getattr(self, "session", None)
        if session is None:
            log.info(
                "voice_output_skipped", session_id=self.session_id, reason="no_livekit_session"
            )
            return
        session.say(text, allow_interruptions=allow_interruptions, add_to_chat_ctx=False)

    async def disconnect(self) -> None:
        session = getattr(self, "session", None)
        if session is not None and hasattr(session, "aclose"):
            await session.aclose()

    def _transcript_text(self, event: Any) -> str:
        if isinstance(event, str):
            return event.strip()
        transcript = getattr(event, "transcript", None)
        if isinstance(transcript, str):
            return transcript.strip()
        alternatives = getattr(event, "alternatives", None) or []
        if alternatives:
            text = getattr(alternatives[0], "text", "") or ""
            return str(text).strip()
        return ""
