"""
Sentiment Analysis FastAPI Server
Uses Hugging Face Inference API with synchronous requests
"""

import logging
import time
import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from contextlib import asynccontextmanager

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

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
    logger.info("=" * 60)
    logger.info("SERVER STARTUP")
    logger.info(f"HF Token configured: {'YES' if HF_API_TOKEN else 'NO'}")
    logger.info("Testing connection to Hugging Face API...")

    try:
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}
        response = requests.get(
            "https://huggingface.co",
            headers=headers,
            timeout=10
        )
        logger.info(f"✅ HF connection test: {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ HF connection test failed: {e}")

    logger.info("✅ Server is ready to receive requests")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down...")


# ============================================================================
# APP
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
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="Server is running and connected to Hugging Face API"
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(request: AnalyzeRequest):
    # Validate
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

        # Build headers
        headers = {
            "Content-Type": "application/json"
        }
        if HF_API_TOKEN:
            headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

        # Call HF API using synchronous requests
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )

        inference_time = time.time() - start_time

        logger.info(f"HF API status: {response.status_code}")

        # Handle 503 (model loading)
        if response.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail="Model is warming up. Please try again in 20 seconds."
            )

        # Handle errors
        if response.status_code != 200:
            logger.error(f"HF API error: {response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Hugging Face API error: {response.status_code}"
            )

        # Parse response
        # HF returns: [[{"label": "POSITIVE", "score": 0.9987}, ...]]
        result = response.json()
        logger.info(f"HF API response: {result}")

        predictions = result[0]
        best = max(predictions, key=lambda x: x["score"])

        logger.info(
            f"✅ {best['label']} | "
            f"{best['score']:.4f} | "
            f"{inference_time*1000:.2f}ms"
        )

        return AnalyzeResponse(
            text=text,
            sentiment=best["label"],
            confidence=best["score"],
            inference_time_ms=round(inference_time * 1000, 2)
        )

    except HTTPException:
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Hugging Face API. Please try again."
        )
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. Please try again."
        )
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
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
