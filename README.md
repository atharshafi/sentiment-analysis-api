# Sentiment Analysis API 🎭

A production-grade **Sentiment Analysis API** built as an AI Engineering learning project. It runs a real transformer neural network (**DistilBERT**) locally for learning, and a lightweight lexicon analyzer (**VADER**) in production — from the exact same codebase.

> **Status:** Live in production ✅ | Dual-engine (DistilBERT + VADER) 🚀

**Live Demo:**
- Frontend: https://sentiment-analysis-api.vercel.app
- Backend API: https://sentiment-analysis-api-backend.onrender.com
- API Docs: https://sentiment-analysis-api-backend.onrender.com/docs

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [The Two-Engine Design](#-the-two-engine-design)
- [The Learning Path (3 Stages)](#-the-learning-path-3-stages)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Stage 0: The Standalone Learning Script](#-stage-0-the-standalone-learning-script)
- [Running Locally (DistilBERT API)](#-running-locally-distilbert-api)
- [How Production Works (VADER)](#-how-production-works-vader)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Learning Outcomes](#-learning-outcomes)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

This project takes a piece of text (e.g. "I love this product!") and classifies its
sentiment as **POSITIVE** or **NEGATIVE**, with a confidence score and the inference
time. It's a full-stack AI application: React frontend → FastAPI backend → sentiment
engine.

**What makes it a great learning project:** it runs two different sentiment engines
from one codebase. Locally you run the real transformer (DistilBERT) so you can learn
the full ML pipeline; in production it automatically uses the lightweight VADER engine
so it fits on a free hosting tier. You switch between them with a single environment
variable — no code changes.

---

## 🧠 The Two-Engine Design

This is the heart of the project. Read this section first.

| | DistilBERT (local) | VADER (hosted) |
|---|---|---|
| **Type** | Transformer neural network | Rule-based lexicon |
| **Accuracy** | ~91% | ~85% |
| **RAM needed** | ~1.5 GB | ~5 MB |
| **Speed** | ~150 ms/request | ~2 ms/request |
| **Install size** | ~2 GB (torch + transformers) | ~50 KB |
| **Best for** | Learning the ML pipeline | Free-tier hosting |
| **Where it runs** | Your laptop | Render free tier |

### Why two engines?

DistilBERT is a genuine transformer model. Running it locally lets you learn the
entire inference pipeline: **tokenization → embeddings → transformer layers → logits
→ softmax → label**. That's the real AI-engineering learning value.

But DistilBERT needs ~1.5 GB of RAM, which crashes Render's 512 MB free tier. VADER
is a rule-based analyzer that runs in ~5 MB, so it's perfect for hosting — same API,
same response shape, no cost.

### How the switch works

A single environment variable, `USE_MODEL`, decides which engine loads at startup:

```
USE_MODEL=distilbert   → loads DistilBERT   (set this locally, in backend/.env)
USE_MODEL=vader        → uses VADER          (default; used on Render)
USE_MODEL not set      → defaults to VADER   (safe fallback for hosting)
```

Because your local `backend/.env` is gitignored, Render never sees it and stays on
VADER automatically. **You change nothing on Render.** The response includes an
`"engine"` field so you can always see which one answered.

---

## 🪜 The Learning Path (3 Stages)

This project is designed to be learned in three stages, from simplest to full-stack:

```
STAGE 0: sentiment_analysis_local_v2.py   → Understand the model in isolation
   (just Python, no server, no frontend — prints results to your terminal)
              │
              ▼
STAGE 1: app.py with USE_MODEL=distilbert  → Wrap DistilBERT in a real API
   (FastAPI server + React frontend, running on your laptop)
              │
              ▼
STAGE 2: app.py with VADER (production)    → Deploy a lightweight version
   (same code, VADER engine, hosted free on Render + Vercel)
```

Each stage builds on the previous one. Start at Stage 0 to *understand* DistilBERT,
then move to Stage 1 to *serve* it, then Stage 2 to *deploy* it.

---

## 🏗️ Architecture

### Local (learning mode - DistilBERT)

```
┌──────────────────────────────────────────────┐
│  Your Browser  →  http://localhost:5174       │
│  React + Vite + Tailwind                      │
│  VITE_API_URL = http://localhost:8000         │
└───────────────────────┬──────────────────────┘
                        │ POST /analyze  {"text": "..."}
                        ▼
┌──────────────────────────────────────────────┐
│  FastAPI  →  http://localhost:8000            │
│  USE_MODEL=distilbert (from backend/.env)     │
│                                               │
│  DistilBERT pipeline (in memory):             │
│   text → tokenize → embeddings → 6 transformer│
│   layers → logits → softmax → POSITIVE/NEG    │
└───────────────────────┬──────────────────────┘
                        │ {"sentiment","confidence","engine":"distilbert"}
                        ▼
              Frontend shows result
```

### Production (hosted mode - VADER)

```
┌──────────────────────────────────────────────┐
│  Any Browser  →  vercel.app                   │
│  React (built + deployed on Vercel CDN)       │
│  VITE_API_URL = ...onrender.com               │
└───────────────────────┬──────────────────────┘
                        │ POST /analyze  {"text": "..."}
                        ▼
┌──────────────────────────────────────────────┐
│  FastAPI  →  ...onrender.com                  │
│  USE_MODEL unset → defaults to VADER          │
│                                               │
│  VADER (in memory, ~5MB):                     │
│   text → lexicon lookup → compound score      │
│   → POSITIVE/NEGATIVE                          │
└───────────────────────┬──────────────────────┘
                        │ {"sentiment","confidence","engine":"vader"}
                        ▼
              Frontend shows result
```

The frontend code is **identical** in both cases. It just calls `/analyze` and
displays the result — it doesn't know or care which engine ran.

---

## 💻 Tech Stack

### Backend
| Tool | Purpose |
|------|---------|
| Python 3.10 | Language |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pydantic | Request/response validation |
| vaderSentiment | Lightweight sentiment engine (hosted) |
| transformers + torch | DistilBERT engine (local only) |

### Frontend
| Tool | Purpose |
|------|---------|
| React 18 | UI framework |
| Vite | Build tool / dev server |
| Tailwind CSS | Styling |

### Infrastructure
| Tool | Purpose |
|------|---------|
| GitHub | Version control + deploy trigger |
| Render | Backend hosting (VADER) |
| Vercel | Frontend hosting |

---

## 📦 Prerequisites

- Python 3.10 (`python --version`)
- Node.js 16+ and npm (`node --version`)
- Git
- ~4 GB free RAM (for DistilBERT locally)
- ~2 GB disk space (torch + model cache)
- Internet (first run downloads the model)

---

## 🔬 Stage 0: The Standalone Learning Script

**File:** `backend/sentiment_analysis_local_v2.py`

### What it is

This is a **standalone learning script** — it is NOT the server and NOT part of the
running app. It loads DistilBERT directly and prints how the model behaves on 20+
example texts (sarcasm, emojis, negations, ALL CAPS, gibberish, and more). Run it
once to *understand* the model before you wrap it in an API.

Think of it as **Stage 0**: understand the model in isolation, with no server and no
frontend to distract you. It runs in your terminal, prints results, and exits.

### What it does NOT do

- ❌ It does not start a web server
- ❌ It does not connect to the frontend
- ❌ It is not imported by `app.py`
- ❌ It is not used when the app is running (local or hosted)

It is purely a teaching/debugging tool.

### How to set it up and run

```bash
# 1. Go to the backend folder
cd backend

# 2. Create and activate a virtual environment (if not already)
python -m venv venv
source venv/bin/activate            # Mac/Linux
# venv\Scripts\Activate.ps1         # Windows PowerShell

# 3. Install the LOCAL dependencies (includes transformers + torch)
pip install -r requirements-local.txt

# 4. Run the script
python sentiment_analysis_local_v2.py
```

### What you'll see

First run downloads DistilBERT (~268 MB, 1–2 minutes). Then it prints:

```
================================================================================
SENTIMENT ANALYSIS - LOCAL MODEL LOADING TEST
================================================================================
Model: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)

RUNNING CORE TEST CASES
Test 1: POSITIVE
Text: I absolutely love this product! It's amazing!
Result: POSITIVE (99.98% confidence)
Inference time: 145.23ms
...

RUNNING EDGE CASE TESTS
Text                                | Result    | Confidence | Notes
-------------------------------------------------------------------------
wow                                 | POSITIVE  | 96.5%      | Very short
Not bad at all!                     | NEGATIVE  | 78.2%      | Negation - tricky!
😊 great!                           | POSITIVE  | 92.1%      | Emoji
...
```

### Why it's worth running

- **See tokenization + inference in action** on real inputs
- **Discover the model's limits** — e.g. it often mislabels sarcasm and negations
- **Understand confidence scores** — ambiguous text lands near 50%
- **Portfolio value** — shows you understand the model, not just an API wrapper

Once you've run this and understood the model, move on to Stage 1 (serving it via an
API).

---

## 🚀 Running Locally (DistilBERT API)

**This is Stage 1.** Here you wrap DistilBERT in a FastAPI server and connect the
React frontend — a full local full-stack app powered by the transformer.

### 1. Clone and enter the project

```bash
git clone https://github.com/atharshafi/sentiment-analysis-api.git
cd sentiment-analysis-api
```

### 2. Backend setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # Mac/Linux
# venv\Scripts\Activate.ps1         # Windows PowerShell

# Install LOCAL dependencies (includes torch + transformers for DistilBERT)
pip install --upgrade pip
pip install -r requirements-local.txt
```

> `requirements-local.txt` includes `transformers` and `torch`.
> The regular `requirements.txt` (used by Render) has only VADER.

### 3. Tell the backend to use DistilBERT

Create `backend/.env`:

```bash
cat > .env << 'EOF'
USE_MODEL=distilbert
EOF
```

Make sure `.env` is gitignored (so Render stays on VADER):

```bash
echo ".env" >> ../.gitignore
```

### 4. Run the backend

```bash
python app.py
```

Expected startup logs:

```
SERVER STARTUP
Selected engine (USE_MODEL): distilbert
Loading DistilBERT model (this may take a moment)...
DistilBERT loaded successfully!
Server is ready to receive requests
```

First run downloads ~268 MB (cached afterwards — later runs work offline).

### 5. Frontend setup (new terminal)

```bash
cd frontend

# Point the frontend at your local backend
cat > .env.local << 'EOF'
VITE_API_URL=http://localhost:8000
EOF

npm install
npm run dev
```

Open **http://localhost:5174**. You're now running the DistilBERT-powered app!

### 6. Verify the engine

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

```json
{
  "text": "I love this!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "inference_time_ms": 145.32,
  "engine": "distilbert"
}
```

The `"engine": "distilbert"` confirms the transformer is running.

---

## 🌐 How Production Works (VADER)

**This is Stage 2.** The hosted app uses VADER automatically — you don't change
anything to make this happen.

**Why VADER in production:** Render's free tier gives 512 MB RAM. DistilBERT needs
~1.5 GB, so it crashes there. VADER runs in ~5 MB and needs no external API, so the
hosted app is fast, free, and reliable.

**How it selects VADER:** On Render, `USE_MODEL` is not set (your local `.env` is
gitignored and never uploaded). The code defaults to VADER when `USE_MODEL` is unset.

### Production deploy configuration

**Render (backend):**
| Setting | Value |
|---|---|
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Env: `PYTHON_VERSION` | `3.10.13` |
| Env: `PYTHONUNBUFFERED` | `1` |
| Env: `USE_MODEL` | *(not set — defaults to VADER)* |

> Render uses `requirements.txt` (VADER only) — NOT `requirements-local.txt`.
> This keeps the deploy small and within the 512 MB RAM limit.

**Vercel (frontend):**
| Setting | Value |
|---|---|
| Framework Preset | Vite |
| Root Directory | `frontend` |
| Env: `VITE_API_URL` | `https://sentiment-analysis-api-backend.onrender.com` |

### Deploy updates

Both Render and Vercel auto-deploy on every push to `main`:

```bash
git add .
git commit -m "your change"
git push origin main
# Render rebuilds backend, Vercel rebuilds frontend — automatically
```

### Confirm production is on VADER

```bash
curl https://sentiment-analysis-api-backend.onrender.com/health
# {"status":"healthy","model_loaded":true,"engine":"vader","message":"..."}
```

---

## 📖 API Reference

Base URL (local): `http://localhost:8000`
Base URL (hosted): `https://sentiment-analysis-api-backend.onrender.com`

### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "engine": "distilbert",
  "message": "Server is running using the distilbert engine"
}
```

### `POST /analyze`
Request:
```json
{ "text": "I love this product!" }
```
Response:
```json
{
  "text": "I love this product!",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "inference_time_ms": 145.32,
  "engine": "distilbert"
}
```

Error responses:
```json
{"detail": "Text cannot be empty"}      // 400
{"detail": "Text too long"}             // 400
{"detail": "No engine loaded"}          // 503
```

### `GET /docs`
Interactive Swagger UI (auto-generated by FastAPI).

---

## 📝 Project Structure

```
sentiment-analysis-api/
├── backend/
│   ├── venv/                          # Virtual environment (gitignored)
│   ├── app.py                         # Dual-engine FastAPI server (Stage 1 & 2)
│   ├── sentiment_analysis_local_v2.py # Standalone learning script (Stage 0)
│   ├── requirements.txt               # PRODUCTION deps (VADER only) - Render uses this
│   ├── requirements-local.txt         # LOCAL deps (+ transformers + torch)
│   ├── Procfile                       # Render start command
│   ├── .env                           # LOCAL only: USE_MODEL=distilbert (gitignored)
│   └── .env.example                   # Template showing USE_MODEL
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── SentimentAnalyzer.jsx  # Main UI component
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── .env.local                     # VITE_API_URL (gitignored)
│   ├── package.json
│   └── vite.config.js
│
├── runtime.txt                        # Python version hint
├── .gitignore
└── README.md                          # This file
```

**Which files matter for each stage:**

| Stage | Key files |
|-------|-----------|
| 0 — Learning script | `sentiment_analysis_local_v2.py`, `requirements-local.txt` |
| 1 — Local API (DistilBERT) | `app.py`, `requirements-local.txt`, `backend/.env` (USE_MODEL=distilbert) |
| 2 — Production (VADER) | `app.py`, `requirements.txt`, `Procfile`, Render env vars |

Same `app.py` powers Stages 1 and 2 — that's the whole point.

---

## 📚 Learning Outcomes

By working through this project you'll understand:

**AI/ML (from Stage 0 script + DistilBERT local mode):**
- What a pre-trained transformer model is
- Tokenization: turning text into numbers
- Embeddings and transformer layers
- Logits → softmax → probabilities → label
- Model download vs. load vs. inference
- Where the model struggles (sarcasm, negation, ambiguity)
- Why model size matters for deployment

**Backend engineering:**
- REST API design with FastAPI
- Request/response validation with Pydantic
- Environment-based configuration (one codebase, two behaviors)
- CORS and how frontend/backend talk across origins
- Graceful fallback (DistilBERT → VADER if RAM runs out)

**Frontend engineering:**
- React hooks and state
- Calling an API with fetch + async/await
- Environment variables in Vite (`VITE_` prefix)

**DevOps:**
- Deploying to Render (backend) and Vercel (frontend)
- Why free-tier constraints (512 MB RAM) drive architecture
- Keeping secrets/config out of Git

---

## 🔧 Troubleshooting

### Stage 0 script: "No module named 'transformers'"
You need the local requirements. Run:
```bash
pip install -r requirements-local.txt
```

### DistilBERT: "Ran out of memory" locally
Your machine needs ~1.5 GB free RAM. Close other apps, or temporarily set
`USE_MODEL=vader` in `backend/.env` to run the light engine.

### First DistilBERT run is very slow
It's downloading the model (~268 MB). Subsequent runs load from cache in a few
seconds and work offline.

### Frontend shows "Cannot reach the API"
- Local: make sure `python app.py` is running and `VITE_API_URL=http://localhost:8000`
- Hosted: the free Render instance sleeps after 15 min; the first request wakes it
  (30–60s). Hit `/health` to wake it.

### Production accidentally trying to use DistilBERT
Check that `.env` is gitignored and `USE_MODEL` is NOT set on Render. Confirm with
`/health` — it should report `"engine": "vader"`.

### Which engine am I running?
Call `/health` or check the `"engine"` field in any `/analyze` response.

---

## 📄 License

MIT License.

---

**Maintainer:** Athar Shafi
**GitHub:** https://github.com/atharshafi/sentiment-analysis-api