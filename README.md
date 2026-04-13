# 🎯 Talent Scout — AI Interview Engine

AI-powered interview engine that conducts adaptive voice interviews, evaluates candidates with a 3-judge LLM ensemble, and generates ranked scorecards.

Built with **LangGraph** · **FastAPI** · **LiveKit** · **Next.js**

---

## ✨ Features

- **🎤 Real-Time Voice Interviews** — LiveKit-powered voice sessions with STT/TTS integration
- **🧠 Adaptive Questioning** — LangGraph state machine selects questions across 4 lanes (technical, behavioral, business case, project deep-dive) based on candidate responses
- **⚖️ 3-Judge Ensemble Evaluation** — Multi-model consensus scoring (Gemini, GPT-4, Claude) with bias detection and flag-on-disagreement
- **📊 Ranked Scorecards** — Weighted dimension scores, strengths/gaps analysis, and recommended actions (advance/hold/reject)
- **👤 Human-in-the-Loop** — Real-time recruiter WebSocket for question injection and score overrides
- **🔒 JWT Authentication** — Role-based access control (candidate, recruiter, system) with graceful dev-mode fallback
- **🛡️ Integrity Monitoring** — Tab-blur detection, copy-paste flags, and per-session audit logging

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Next.js   │────▶│   FastAPI    │────▶│  LangGraph  │
│  Frontend   │ WS  │  API Layer   │     │  Interview  │
└─────────────┘     └──────┬───────┘     │   Engine    │
                           │             └──────┬──────┘
                    ┌──────┼──────┐             │
                    │      │      │        ┌────▼────┐
               ┌────▼─┐ ┌─▼──┐ ┌─▼───┐   │ 3-Judge │
               │Postgres│ │Redis│ │Chroma│   │Ensemble │
               └───────┘ └────┘ └─────┘   └─────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- Redis 7+
- Node.js 20+ (for frontend)

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/talent-scout.git
cd talent-scout
```

```bash
# Backend
pip install uv
uv pip install -e ".[dev]"

# Frontend
cd frontend && npm install && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set at minimum:
#   GEMINI_API_KEY or OPENROUTER_API_KEY (at least one required)
#   POSTGRES_URL, REDIS_URL, LIVEKIT_URL
```

### 3. Start infrastructure

```bash
docker compose up -d   # Postgres, Redis, ChromaDB, LiveKit
```

### 4. Run migrations

```bash
alembic upgrade head
```

### 5. Launch

```bash
# Backend (port 8001)
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Frontend (port 3000)
cd frontend && npm run dev
```

## 📁 Project Structure

```
├── app/
│   ├── core/           # Config, DB, auth, LLM client, ChromaDB
│   ├── models.py       # SQLModel/Pydantic data models
│   ├── pipeline/       # JD parsing, interview graph, evaluation
│   ├── routes/         # API sub-routers (candidate, recruiter, system)
│   └── main.py         # FastAPI app entry point
├── frontend/           # Next.js candidate & recruiter UI
├── tests/              # Pytest suite (68 tests)
├── alembic/            # Database migrations
├── Dockerfile          # Production container
└── .github/workflows/  # CI/CD pipeline
```

## 🔧 Configuration

Settings are loaded from environment variables. See [`.env.example`](.env.example) for all options.

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | One of these | Google Gemini API key |
| `OPENROUTER_API_KEY` | required | OpenRouter API key |
| `POSTGRES_URL` | ✅ | PostgreSQL connection string |
| `REDIS_URL` | ✅ | Redis connection string |
| `LIVEKIT_URL` | ✅ | LiveKit server URL |
| `JWT_SECRET_KEY` | Production | Enables auth (empty = dev mode) |
| `CORS_ALLOWED_ORIGINS` | Production | Allowed frontend origins |

## 🧪 LLM Providers

The application supports two provider paths, selected automatically per model name:

- **Gemini (default):** Free-tier stack (`gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-embedding-001`). Set `GEMINI_API_KEY`.
- **OpenRouter:** Paid multi-model routing (`openai/gpt-4.1`, `anthropic/claude-sonnet-4-5`, etc.). Set `OPENROUTER_API_KEY`.

At least one provider key must be set.

## 🧪 Testing

```bash
pytest tests/ -v --cov=app
```

68 tests covering pipeline logic, evaluation, API endpoints, and WebSocket flows.

## 📋 API Endpoints

### Candidate
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions/{id}/info` | Session metadata |
| `POST` | `/api/sessions/{id}/complete` | Submit completed interview |
| `POST` | `/api/sessions/{id}/integrity_flag` | Report integrity flag |
| `POST` | `/api/livekit/token` | Get LiveKit room token |
| `WS` | `/api/ws/interviews/{id}` | Real-time interview |

### Recruiter
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/intake/batch` | Submit job + candidates |
| `GET` | `/api/intake/status/{job_id}` | Session status counts |
| `GET` | `/api/recruiter/{job_id}/scorecards` | All scorecards |
| `GET` | `/api/recruiter/{job_id}/scorecard/{id}` | Scorecard detail |
| `POST` | `/api/recruiter/{job_id}/override` | Override dimension score |
| `POST` | `/api/recruiter/{job_id}/finalize` | Finalize shortlist |
| `WS` | `/api/ws/recruiter/{job_id}` | Real-time HITL control |

### System
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/auth/token` | Generate dev JWT |

## 📝 License

MIT

---

Built by Eslam
