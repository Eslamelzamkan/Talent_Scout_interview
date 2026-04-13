from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Authentication ---
    jwt_secret_key: str | None = None
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440

    # --- CORS ---
    cors_allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # --- LLM Providers ---
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    model_interview: str = "gemini-2.0-flash"
    model_judge_1: str = "gemini-2.0-flash"
    model_judge_2: str = "gemini-2.5-flash"
    model_judge_3: str = "gemini-2.5-flash-lite"
    model_fast: str = "gemini-2.0-flash"
    model_embedding: str = "gemini-embedding-001"

    # --- Infrastructure ---
    postgres_url: str
    redis_url: str
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout: int = 30

    # --- Voice ---
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    interview_voice_id: str | None = None
    google_stt_credentials_json: str | None = None
    cartesia_api_key: str | None = None
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_stt_model: str = "whisper-large-v3-turbo"
    edge_tts_voice: str = "en-US-GuyNeural"

    # --- Observability ---
    langsmith_api_key: str | None = None
    langsmith_project: str = "talent-scout-dev"

    # --- Scoring ---
    interview_weight: float = 0.60
    assessment_weight: float = 0.30
    speech_weight: float = 0.05
    screening_weight: float = 0.05
    advance_threshold: float = 0.75
    hold_threshold: float = 0.50
    max_questions: int = 10
    rpm_limit_gpt4: int = 500
    rpm_limit_claude: int = 400
    rpm_limit_gemini: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_weights(self) -> Settings:
        if not self.openrouter_api_key and not self.gemini_api_key:
            raise ValueError("set either OPENROUTER_API_KEY or GEMINI_API_KEY")
        total = (
            self.interview_weight
            + self.assessment_weight
            + self.speech_weight
            + self.screening_weight
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError("pipeline weights must sum to 1.0")
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @property
    def openrouter_headers(self) -> dict[str, str]:
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when using OpenRouter models")
        return {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://talentscout.ai",
            "X-Title": "Talent Scout",
        }

    @property
    def gemini_headers(self) -> dict[str, str]:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when using Gemini models")
        return {"Authorization": f"Bearer {self.gemini_api_key}"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
