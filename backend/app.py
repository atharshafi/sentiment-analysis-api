"""
Sentiment Analysis FastAPI Server
Uses Hugging Face Inference API (no local model needed!)
Memory usage: ~50MB (vs 1.5GB with local model)
"""

import logging
import time
import os
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HUGGING FACE API CONFIG
# ============================================================================
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for sentiment"
    )
    max_length: Optional[int] = Field(512, description="Max tokens")


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
    logger.info("=" * 60)
    logger.info("SERVER STARTUP")
    logger.info("=" * 60)
    logger.info("Using Hugging Face Inference API")
    logger.info(f"Model: distilbert-base-uncased-finetuned-sst-2-english")
    logger.info(f"HF Token configured: {'YES' if HF_API_TOKEN else 'NO (using free tier)'}")
    logger.info("✅ Server is ready to receive requests")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down...")


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment using Hugging Face Inference API",
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

logger.info("✅ CORS configured for React frontend")


# ============================================================================
# ROUTES
# ============================================================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="Server is running and connected to Hugging Face API"
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sentiment(request: AnalyzeRequest):
    # Validate input
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty or whitespace only"
        )

    if len(text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Text exceeds maximum length of 10,000 characters"
        )

    logger.info(f"Analyzing: {text[:50]}...")

    try:
        start_time = time.time()

        # Call Hugging Face Inference API
        headers = {}
        if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": text}
            )

        inference_time = time.time() - start_time

        # Handle model loading (HF API returns 503 when warming up)
        if response.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail="Model is warming up on Hugging Face servers. Please try again in 20 seconds."
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Hugging Face API error: {response.text}"
            )

        # Parse response
        # HF returns: [[{"label": "POSITIVE", "score": 0.9987}, {...}]]
        result = response.json()
        predictions = result[0]

        # Get the highest confidence prediction
        best_prediction = max(predictions, key=lambda x: x["score"])

        sentiment = best_prediction["label"]
        confidence = best_prediction["score"]

        logger.info(
            f"✅ Result: {sentiment} | "
            f"Confidence: {confidence:.4f} | "
            f"Time: {inference_time*1000:.2f}ms"
        )

        return AnalyzeResponse(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            inference_time_ms=round(inference_time * 1000, 2)
        )

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Request to Hugging Face API timed out. Please try again."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/")
async def root():
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "description": "Analyze sentiment using Hugging Face Inference API",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs"
        }
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
