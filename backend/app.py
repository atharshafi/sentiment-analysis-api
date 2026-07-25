"""
Sentiment Analysis FastAPI Server
==================================

Dual-engine sentiment analysis:
  - DistilBERT (local development): ~91% accuracy, full ML pipeline, ~1.5GB RAM
  - VADER (production/hosted):       ~85% accuracy, lightweight, ~5MB RAM

The engine is selected by the USE_MODEL environment variable:
  USE_MODEL=distilbert  -> loads DistilBERT via transformers (local)
  USE_MODEL=vader       -> uses VADER (default, used on Render)

If USE_MODEL is not set, it defaults to VADER (safe for free-tier hosting).

Why two engines?
  DistilBERT is a real transformer neural network. Running it locally lets you
  learn the full pipeline: tokenization -> embeddings -> transformer layers ->
  logits -> softmax. But it needs ~1.5GB RAM, which crashes Render's 512MB free
  tier. VADER is a rule-based lexicon analyzer that runs in ~5MB, so it's perfect
  for hosting. Both return the exact same response shape (plus an "engine" field),
  so the frontend never needs to know which one is running.
"""

import logging
import time
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load backend/.env if present (local development).
# On Render there is no .env file, so this does nothing there.
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG - WHICH ENGINE TO USE
# ============================================================================
# Read the engine choice from environment.
#   Local machine   -> set USE_MODEL=distilbert in backend/.env
#   Render (hosted) -> leave unset (defaults to vader) or set USE_MODEL=vader
USE_MODEL = os.getenv("USE_MODEL", "vader").lower().strip()

# These globals are populated on startup depending on the engine.
vader_analyzer = None       # VADER analyzer instance
distilbert_pipeline = None  # DistilBERT transformers pipeline

MODEL_NAME_DISTILBERT = "distilbert-base-uncased-finetuned-sst-2-english"


# ============================================================================
# PYDANTIC MODELS (request / response shapes)
# ============================================================================
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    max_length: Optional[int] = Field(512)


class AnalyzeResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    inference_time_ms: float
    engine: str  # "distilbert" or "vader" - tells you which engine ran


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    engine: str
    message: str


# ============================================================================
# ENGINE LOADERS
# ============================================================================
def load_vader():
    """Load the VADER analyzer (tiny, instant)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    logger.info("Loading VADER sentiment analyzer...")
    analyzer = SentimentIntensityAnalyzer()
    logger.info("VADER loaded successfully!")
    return analyzer


def load_distilbert():
    """
    Load the DistilBERT model via the transformers pipeline.

    First run downloads ~268MB from Hugging Face and caches it in
    ~/.cache/huggingface/. Subsequent runs load from cache (works offline).
    device=-1 forces CPU (no GPU required).
    """
    from transformers import pipeline
    logger.info("Loading DistilBERT model (this may take a moment)...")
    logger.info(f"Model: {MODEL_NAME_DISTILBERT}")
    clf = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME_DISTILBERT,
        device=-1,  # -1 = CPU
    )
    logger.info("DistilBERT loaded successfully!")
    return clf


# ============================================================================
# LIFESPAN - runs on startup / shutdown
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vader_analyzer, distilbert_pipeline

    logger.info("=" * 60)
    logger.info("SERVER STARTUP")
    logger.info(f"Selected engine (USE_MODEL): {USE_MODEL}")

    if USE_MODEL == "distilbert":
        # Load the transformer model for local learning/development
        try:
            distilbert_pipeline = load_distilbert()
        except Exception as e:
            # If DistilBERT fails (e.g. no RAM), fall back to VADER so the
            # server still starts instead of crashing.
            logger.error(f"Failed to load DistilBERT: {e}")
            logger.warning("Falling back to VADER.")
            vader_analyzer = load_vader()
    else:
        # Default: VADER (used on Render / hosted)
        vader_analyzer = load_vader()

    logger.info("Server is ready to receive requests")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down...")


# ============================================================================
# APP
# ============================================================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="Dual-engine sentiment analysis (DistilBERT local / VADER hosted)",
    version="2.0.0",
    lifespan=lifespan,
)

# ============================================================================
# CORS
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)
logger.info("CORS configured")


# ============================================================================
# HELPER: which engine is actually active right now?
# ============================================================================
def active_engine() -> str:
    if distilbert_pipeline is not None:
        return "distilbert"
    if vader_analyzer is not None:
        return "vader"
    return "none"


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================
def analyze_with_distilbert(text: str):
    """
    Run DistilBERT inference.

    The pipeline returns e.g. [{"label": "POSITIVE", "score": 0.9998}].
    DistilBERT only outputs POSITIVE or NEGATIVE (no explicit NEUTRAL).
    """
    result = distilbert_pipeline(text, truncation=True, max_length=512)
    best = result[0]
    return best["label"], float(best["score"])


def analyze_with_vader(text: str):
    """
    Run VADER analysis.

    VADER returns a compound score from -1 (very negative) to +1 (very positive).
    We map it to POSITIVE/NEGATIVE and normalize confidence to 0.5-1.0.
    """
    scores = vader_analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "POSITIVE"
        confidence = (compound + 1) / 2
    elif compound <= -0.05:
        sentiment = "NEGATIVE"
        confidence = (-compound + 1) / 2
    else:
        # Neutral zone: pick whichever side is stronger
        if scores["pos"] >= scores["neg"]:
            sentiment = "POSITIVE"
            confidence = 0.5 + scores["pos"] * 0.1
        else:
            sentiment = "NEGATIVE"
            confidence = 0.5 + scores["neg"] * 0.1

    confidence = min(max(confidence, 0.5), 1.0)
    return sentiment, confidence


# ============================================================================
# ROUTES
# ============================================================================
@app.get("/")
def root():
    return {
        "name": "Sentiment Analysis API",
        "version": "2.0.0",
        "status": "running",
        "engine": active_engine(),
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    engine = active_engine()
    return HealthResponse(
        status="healthy",
        model_loaded=(engine != "none"),
        engine=engine,
        message=f"Server is running using the {engine} engine",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(request: AnalyzeRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long")

    engine = active_engine()
    if engine == "none":
        raise HTTPException(status_code=503, detail="No engine loaded")

    logger.info(f"[{engine}] Analyzing: {text[:50]}...")

    try:
        start_time = time.time()

        if engine == "distilbert":
            sentiment, confidence = analyze_with_distilbert(text)
        else:
            sentiment, confidence = analyze_with_vader(text)

        inference_time = time.time() - start_time

        logger.info(
            f"[{engine}] {sentiment} | "
            f"{confidence:.4f} | {inference_time*1000:.2f}ms"
        )

        return AnalyzeResponse(
            text=text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            inference_time_ms=round(inference_time * 1000, 2),
            engine=engine,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)