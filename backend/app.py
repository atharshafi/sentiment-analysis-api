"""
Sentiment Analysis FastAPI Server
Uses VADER - runs locally, no external API needed!
Perfect for Render free tier (no memory/DNS issues)
"""

import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL ANALYZER
# ============================================================================
analyzer = None

# ============================================================================
# MODELS
# ============================================================================
class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    max_length: Optional[int] = Field(512)

class AnalyzeResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str

# ============================================================================
# LIFESPAN
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("=" * 60)
    logger.info("SERVER STARTUP")
    logger.info("Loading VADER sentiment analyzer...")

    analyzer = SentimentIntensityAnalyzer()

    logger.info("✅ VADER loaded successfully!")
    logger.info("✅ Server is ready to receive requests")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down...")

# ============================================================================
# APP
# ============================================================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment using VADER",
    version="1.0.0",
    lifespan=lifespan
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
logger.info("✅ CORS configured")

# ============================================================================
# ROUTES
# ============================================================================
@app.get("/")
def root():
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "status": "running",
        "model": "VADER Sentiment Analyzer"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=analyzer is not None,
        message="Server is running with VADER sentiment analyzer"
    )

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(request: AnalyzeRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long")

    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not loaded")

    logger.info(f"Analyzing: {text[:50]}...")

    try:
        start_time = time.time()

        # Run VADER sentiment analysis
        scores = analyzer.polarity_scores(text)

        inference_time = time.time() - start_time

        # VADER returns:
        # {
        #   'neg': 0.0,
        #   'neu': 0.295,
        #   'pos': 0.705,
        #   'compound': 0.8316
        # }
        # compound score: -1 (most negative) to +1 (most positive)
        # >= 0.05 = POSITIVE
        # <= -0.05 = NEGATIVE
        # between = NEUTRAL (we map to NEGATIVE or POSITIVE)

        compound = scores['compound']
        pos = scores['pos']
        neg = scores['neg']

        if compound >= 0.05:
            sentiment = "POSITIVE"
            confidence = (compound + 1) / 2  # normalize to 0-1
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
            confidence = (-compound + 1) / 2  # normalize to 0-1
        else:
            # Neutral - pick whichever is stronger
            if pos >= neg:
                sentiment = "POSITIVE"
                confidence = 0.5 + (pos * 0.1)
            else:
                sentiment = "NEGATIVE"
                confidence = 0.5 + (neg * 0.1)

        # Ensure confidence is between 0.5 and 1.0
        confidence = min(max(confidence, 0.5), 1.0)

        logger.info(
            f"✅ {sentiment} | "
            f"confidence: {confidence:.4f} | "
            f"compound: {compound:.4f} | "
            f"{inference_time*1000:.2f}ms"
        )

        return AnalyzeResponse(
            text=text,
            sentiment=sentiment,
            confidence=round(confidence, 4),
            inference_time_ms=round(inference_time * 1000, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
