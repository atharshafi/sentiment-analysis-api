# Sentiment Analysis API 🎭

A production-grade **Sentiment Analysis API** built as an AI-Engineering learning project. It runs a real transformer neural network (**DistilBERT**) locally for learning, and a lightweight lexicon analyzer (**VADER**) in production — from the exact same codebase, switched by a single environment variable.

> **Status:** Live in production ✅ | Dual-engine (DistilBERT local / VADER hosted) 🚀

**Live Demo**
- Frontend: https://sentiment-analysis-api.vercel.app
- Backend API: https://sentiment-analysis-api-backend.onrender.com
- API Docs (Swagger): https://sentiment-analysis-api-backend.onrender.com/docs

---

## 📋 Table of Contents

1. [What This Project Does](#-what-this-project-does)
2. [The Architecture We Followed (and Why)](#-the-architecture-we-followed-and-why)
3. [The Two-Engine Design](#-the-two-engine-design)
4. [The 3-Stage Learning Path](#-the-3-stage-learning-path)
5. [System Architecture Diagrams](#-system-architecture-diagrams)
6. [Tech Stack](#-tech-stack)
7. [Prerequisites](#-prerequisites)
8. [Stage 0 — The Standalone Learning Script](#-stage-0--the-standalone-learning-script)
9. [Setting Up Locally (DistilBERT)](#-setting-up-locally-distilbert)
10. [Setting Up for Production (VADER)](#-setting-up-for-production-vader)
11. [API Reference](#-api-reference)
12. [Project Structure](#-project-structure)
13. [Environment Variables Reference](#-environment-variables-reference)
14. [Security Notes](#-security-notes)
15. [Learning Outcomes](#-learning-outcomes)
16. [Troubleshooting](#-troubleshooting)

---

## 🎯 What This Project Does

Give it a sentence, get back its sentiment.

**Input:** `"I love this product!"`
**Output:** `POSITIVE` with a confidence score and how long the analysis took.

It's a full-stack AI application:

```
React frontend  →  FastAPI backend  →  sentiment engine  →  result
```

The interesting part is the backend. It can run **two completely different sentiment
engines** from one codebase — a heavy, accurate transformer for local learning, and a
tiny, fast lexicon analyzer for free hosting. You choose which one with a single
environment variable, changing zero lines of code.

---

## 🏛️ The Architecture We Followed (and Why)

We followed a **decoupled, environment-configurable, stateless microservice**
architecture. That's a mouthful — here's what each part means and why we chose it.

### 1. Decoupled Frontend & Backend

The React frontend and FastAPI backend are **separate applications**, deployed
independently (Vercel for frontend, Render for backend). They communicate only over
HTTP/JSON.

**Why:**
- Each can be developed, deployed, and scaled on its own
- The frontend is a static site (cheap, global CDN, instant loads)
- The backend can change engines without the frontend knowing or caring
- Mirrors how real companies structure production apps

### 2. Environment-Configurable Engine Selection

Instead of hardcoding which sentiment engine to use, the backend reads an environment
variable (`USE_MODEL`) at startup and loads the matching engine.

**Why:**
- One codebase behaves differently in different environments
- No separate branches or duplicate files to maintain
- Local = DistilBERT (learning), Production = VADER (hosting) — automatically
- This is the "12-Factor App" principle: *config lives in the environment, not the code*

### 3. Stateless API (No Database)

Every request is fully independent. Text comes in, sentiment goes out, nothing is
stored.

**Why:**
- Simplicity — no database to manage, secure, or pay for
- Scales trivially — any server can handle any request
- Fast — no DB round-trips
- Privacy — user text is analyzed and immediately discarded

### 4. Graceful Degradation

If DistilBERT fails to load (e.g. not enough RAM), the server automatically falls back
to VADER instead of crashing.

**Why:**
- The server always starts and stays useful
- Protects against out-of-memory failures on constrained machines

### Why not just run DistilBERT everywhere?

Because DistilBERT needs **~1.5 GB of RAM**, and free hosting (Render) gives only
**512 MB**. Loading DistilBERT there crashes the server. VADER runs in **~5 MB**, so
it's perfect for hosting. Rather than give up the transformer entirely, we use it where
we have the resources (local) and swap to VADER where we don't (hosted). Same API, same
response shape — the frontend never notices.

---

## 🧠 The Two-Engine Design

| | DistilBERT (local) | VADER (hosted) |
|---|---|---|
| **Type** | Transformer neural network | Rule-based lexicon |
| **Accuracy** | ~91% | ~85% |
| **RAM needed** | ~1.5 GB | ~5 MB |
| **Speed** | ~150 ms/request | ~2 ms/request |
| **Install size** | ~2 GB (torch + transformers) | ~50 KB |
| **Handles negation/sarcasm** | Better | Weaker |
| **Best for** | Learning the ML pipeline | Free-tier hosting |
| **Where it runs** | Your laptop | Render free tier |

### How the switch works

A single environment variable decides the engine at startup:

```
USE_MODEL=distilbert   → loads DistilBERT   (set locally in backend/.env)
USE_MODEL=vader        → uses VADER          (default; used on Render)
USE_MODEL not set      → defaults to VADER   (safe fallback for hosting)
```

Because `backend/.env` is **gitignored**, it never reaches Render, so production stays
on VADER automatically. **You change nothing on Render.** Every `/analyze` response
includes an `"engine"` field so you can always confirm which one answered.

---

## 🪜 The 3-Stage Learning Path

This project is meant to be learned in three stages, simplest to full-stack:

```
STAGE 0 — Understand the model in isolation
   File: backend/sentiment_analysis_local_v2.py
   Just Python. No server, no frontend. Prints DistilBERT results to the terminal.
              │
              ▼
STAGE 1 — Serve the model via an API (local full-stack)
   File: backend/app.py with USE_MODEL=distilbert
   FastAPI server + React frontend, DistilBERT engine, on your laptop.
              │
              ▼
STAGE 2 — Deploy a lightweight version (production)
   File: backend/app.py with VADER (default)
   Same code, VADER engine, hosted free on Render + Vercel.
```

Start at Stage 0 to *understand* DistilBERT, move to Stage 1 to *serve* it, then
Stage 2 to *deploy* it.

---

## 🏗️ System Architecture Diagrams

### Local (learning mode — DistilBERT)

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

### Production (hosted mode — VADER)

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
displays the result — it never needs to know which engine ran.

---

## 💻 Tech Stack

### Backend
| Tool | Purpose |
|------|---------|
| Python 3.11 (local) / 3.10 (Render) | Language |
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pydantic | Request/response validation |
| python-dotenv | Loads local `.env` file |
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

- **Python 3.10 or 3.11** — NOT 3.14 (too new; torch/tokenizers have no wheels yet)
- Node.js 16+ and npm
- Git
- ~4 GB free RAM (for DistilBERT locally)
- ~2 GB disk space (torch + model cache)
- Internet (first run downloads the model)

> ⚠️ **Important:** If your default `python` is 3.14, create the venv with
> `python3.11` explicitly (see setup below). Python 3.14 will fail to install torch,
> tokenizers, and pydantic-core because prebuilt wheels don't exist for it yet.

---

## 🔬 Stage 0 — The Standalone Learning Script

**File:** `backend/sentiment_analysis_local_v2.py`

### What it is

A **standalone learning script** — NOT the server, NOT part of the running app. It
loads DistilBERT directly and prints how the model behaves on 20+ example texts
(sarcasm, emojis, negations, ALL CAPS, gibberish, and more). Run it once to
*understand* the model before wrapping it in an API.

### What it does NOT do

- ❌ Does not start a web server
- ❌ Does not connect to the frontend
- ❌ Is not imported by `app.py`
- ❌ Is not used when the app is running (local or hosted)

It is purely a teaching/debugging tool — **Stage 0** of the learning path.

### How to run it

```bash
cd backend

# Create + activate a Python 3.11 virtual environment (see note in Prerequisites)
python3.11 -m venv venv
source venv/bin/activate            # Mac/Linux
# venv\Scripts\Activate.ps1         # Windows PowerShell

# Install local dependencies (includes transformers + torch)
pip install -r requirements-local.txt

# Run the script
python sentiment_analysis_local_v2.py
```

### What you'll see

First run downloads DistilBERT (~268 MB, 1–2 min). Then:

```
================================================================================
SENTIMENT ANALYSIS - LOCAL MODEL LOADING TEST
================================================================================
Test 1: POSITIVE
Text: I absolutely love this product! It's amazing!
Result: POSITIVE (99.98% confidence)
Inference time: 145.23ms
...
Text                                | Result    | Confidence | Notes
-------------------------------------------------------------------------
Not bad at all!                     | NEGATIVE  | 78.2%      | Negation - tricky!
😊 great!                           | POSITIVE  | 92.1%      | Emoji
```

### Why it's worth running

- See tokenization + inference on real inputs
- Discover where the model struggles (sarcasm, negation)
- Understand confidence scores (ambiguous text lands near 50%)
- Portfolio value — proves you understand the model, not just an API wrapper

Once you've run this, move on to Stage 1.

---

## 🚀 Setting Up Locally (DistilBERT)

**This is Stage 1** — wrap DistilBERT in a FastAPI server and connect the React
frontend. A full local full-stack app powered by the transformer.

### 1. Clone the repo

```bash
git clone https://github.com/atharshafi/sentiment-analysis-api.git
cd sentiment-analysis-api
```

### 2. Backend setup

```bash
cd backend

# IMPORTANT: use Python 3.10 or 3.11, NOT 3.14
python3.11 -m venv venv
source venv/bin/activate            # Mac/Linux
# venv\Scripts\Activate.ps1         # Windows PowerShell

# Verify the version (must NOT be 3.14)
python --version                    # e.g. Python 3.11.x

# Install LOCAL dependencies (includes torch + transformers)
pip install --upgrade pip
pip install -r requirements-local.txt
```

> `requirements-local.txt` includes `transformers` + `torch` for DistilBERT.
> The plain `requirements.txt` (used by Render) has only VADER.

### 3. Tell the backend to use DistilBERT

Create `backend/.env`:

```bash
cat > .env << 'EOF'
USE_MODEL=distilbert
EOF
```

Confirm `.env` is gitignored (so Render never sees it and stays on VADER):

```bash
grep -q "\.env" ../.gitignore || echo ".env" >> ../.gitignore
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

> You may see `FutureWarning` messages from torch/transformers — these are harmless
> version-deprecation notices and don't affect functionality.

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

`"engine": "distilbert"` confirms the transformer is running locally.

---

## 🌐 Setting Up for Production (VADER)

**This is Stage 2.** The hosted app uses VADER automatically — you don't change any
code to make this happen.

### Why VADER in production

Render's free tier gives 512 MB RAM. DistilBERT needs ~1.5 GB, so it crashes there.
VADER runs in ~5 MB and needs no external API, so the hosted app is fast, free, and
reliable.

### How it selects VADER

On Render, `USE_MODEL` is not set (your local `.env` is gitignored and never uploaded).
The code defaults to VADER when `USE_MODEL` is unset.

### Backend deploy (Render)

| Setting | Value |
|---|---|
| Build Command | `pip install -r backend/requirements.txt` |
| Start Command | `cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Env: `PYTHON_VERSION` | `3.10.13` |
| Env: `PYTHONUNBUFFERED` | `1` |
| Env: `USE_MODEL` | *(not set — defaults to VADER)* |

> ⚠️ Render must install `requirements.txt` (VADER only), NOT `requirements-local.txt`.
> If torch/transformers get installed on Render, it will run out of memory.

### Frontend deploy (Vercel)

| Setting | Value |
|---|---|
| Framework Preset | Vite |
| Root Directory | `frontend` |
| Env: `VITE_API_URL` | `https://sentiment-analysis-api-backend.onrender.com` |

### Pushing updates (auto-deploy)

Both Render and Vercel redeploy automatically on every push to `main`:

```bash
git add .
git commit -m "your change"
git push origin main
```

### Before you push — safety checklist

```bash
# 1. Make sure .env is NOT going to be committed
git status | grep "\.env$"        # should print nothing (or only .env.example)

# 2. Make sure requirements.txt is VADER-only (no torch/transformers)
cat backend/requirements.txt
```

If `backend/.env` shows up in `git status`, add it to `.gitignore` before pushing —
otherwise Render will try to load DistilBERT and crash.

### Confirm production is on VADER

```bash
curl https://sentiment-analysis-api-backend.onrender.com/health
# {"status":"healthy","model_loaded":true,"engine":"vader","message":"..."}
```

`"engine":"vader"` = production is safe and lightweight.

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
Errors:
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
│   ├── requirements.txt               # PRODUCTION deps (VADER only) — Render uses this
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

The same `app.py` powers Stages 1 and 2 — that's the core idea.

---

## 🔐 Environment Variables Reference

### Backend

| Variable | Where | Value | Purpose |
|----------|-------|-------|---------|
| `USE_MODEL` | `backend/.env` (local) | `distilbert` | Selects DistilBERT locally |
| `USE_MODEL` | Render (hosted) | *(unset)* | Defaults to VADER |
| `PYTHON_VERSION` | Render | `3.10.13` | Stable Python with prebuilt wheels |
| `PYTHONUNBUFFERED` | Render | `1` | Real-time logs |

### Frontend

| Variable | Where | Value | Purpose |
|----------|-------|-------|---------|
| `VITE_API_URL` | `frontend/.env.local` (local) | `http://localhost:8000` | Points to local backend |
| `VITE_API_URL` | Vercel (hosted) | `https://...onrender.com` | Points to live backend |

> Vite requires the `VITE_` prefix. `REACT_APP_` (Create React App style) does NOT work
> with Vite — it silently fails and falls back to localhost.

---

## 🔒 Security Notes

**Safe to be public:**
- ✅ The API endpoint URL — it's a public API, like a restaurant's phone number
- ✅ The GitHub repository — open-source portfolio project
- ✅ The frontend code — contains no secrets

**Never committed (gitignored):**
- ❌ `backend/.env` — local engine config
- ❌ `frontend/.env.local` — local API URL
- ❌ `backend/venv/` — virtual environment

**No secrets in this project:** Since production uses VADER (no external API, no token,
no database), there are no API keys or passwords to protect. If you later add a
paid model or database, keep those credentials in environment variables only — never
in code.

**Is displaying the endpoint in the UI safe?** Yes. The API requires no authentication,
stores no data, and has no usage cost (VADER is free and unlimited). Showing
`.../analyze` in the frontend is completely fine.

---

## 📚 Learning Outcomes

**AI/ML (from Stage 0 script + DistilBERT local mode):**
- What a pre-trained transformer model is
- Tokenization: turning text into numbers
- Embeddings, transformer layers, logits → softmax → label
- Model download vs. load vs. inference
- Where models struggle (sarcasm, negation, ambiguity)
- Why model size drives deployment decisions

**Backend engineering:**
- REST API design with FastAPI
- Request/response validation with Pydantic
- Environment-based configuration (one codebase, two behaviors)
- CORS and cross-origin communication
- Graceful fallback (DistilBERT → VADER if RAM runs out)

**Frontend engineering:**
- React hooks and state management
- Calling an API with fetch + async/await
- Vite environment variables (`VITE_` prefix)

**DevOps:**
- Deploying to Render (backend) and Vercel (frontend)
- Why free-tier constraints (512 MB RAM) drive architecture
- Keeping config/secrets out of Git
- Matching local Python version to production (3.10/3.11)

---

## 🔧 Troubleshooting

### Install fails with "maturin" / Rust / "Failed building wheel for tokenizers"
Your venv is using **Python 3.14**, which has no prebuilt wheels. Recreate the venv with
3.11:
```bash
deactivate; rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
python --version           # must be 3.11.x, not 3.14
pip install -r requirements-local.txt
```
No Python 3.11? Install it: `brew install python@3.11` (Mac).

### "torch==X.X.X not found"
PyPI removed the pinned version. Update `requirements-local.txt` to the current torch
(e.g. `torch==2.12.1`) and reinstall.

### Server says "Loading VADER" locally but I want DistilBERT
Your `app.py` is the old VADER-only version, OR `backend/.env` is missing, OR `app.py`
lacks `load_dotenv()`. Ensure:
- `app.py` is the dual-engine version (reads `USE_MODEL`)
- `backend/.env` contains `USE_MODEL=distilbert`
- `app.py` calls `load_dotenv()` near the top

### DistilBERT: "Ran out of memory" locally
You need ~1.5 GB free RAM. Close other apps, or temporarily set `USE_MODEL=vader` in
`backend/.env`.

### First DistilBERT run is very slow
It's downloading the model (~268 MB). Later runs load from cache in seconds and work
offline.

### Frontend shows "Cannot reach the API"
- Local: ensure `python app.py` is running and `VITE_API_URL=http://localhost:8000`
- Hosted: the free Render instance sleeps after 15 min; the first request wakes it
  (30–60s). Hit `/health` to wake it.

### Production accidentally trying to use DistilBERT
Confirm `backend/.env` is gitignored, `USE_MODEL` is NOT set on Render, and
`requirements.txt` has no torch/transformers. Verify with `/health` →
`"engine":"vader"`.

### Which engine am I running?
Call `/health`, or check the `"engine"` field in any `/analyze` response.

---

## 📄 License

MIT License.

---

**Maintainer:** Athar Shafi
**GitHub:** https://github.com/atharshafi/sentiment-analysis-api