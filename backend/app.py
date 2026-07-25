"""
Sentiment Analysis FastAPI Server
Uses Hugging Face Router API (new endpoint - works on Render free tier)
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
# New router URL - works on Render free tier!
HF_API_URL = "https://router.huggingface.co/hf-inference/models/distilbert-base-uncased-finetuned-sst-2-english/v1/text-classification"
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
    logger.info(f"HF Token: {'SET ✅' if HF_API_TOKEN else 'NOT SET ❌'}")
    logger.info(f"API URL: {HF_API_URL}")

    # Test connection to router (not api-inference!)
    try:
        test_response = requests.get(
            "https://router.huggingface.co",
            timeout=10
        )
        logger.info(f"✅ Router connection test: {test_response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Router connection test failed: {e}")

    logger.info("✅ Server is ready to receive requests")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down...")

# ============================================================================
# APP
# ============================================================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment using Hugging Face Router API",
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
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="Server is running and connected to Hugging Face Router API"
    )

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_sentiment(request: AnalyzeRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long")

    logger.info(f"Analyzing: {text[:50]}...")

    try:
        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HF_API_TOKEN}"
        }

        # New router API format
        payload = {"inputs": text}

        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        inference_time = time.time() - start_time

        logger.info(f"Router API status: {response.status_code}")
        logger.info(f"Router API response: {response.text[:200]}")

        if response.status_code == 503:
            raise HTTPException(
                status_code=503,
                detail="Model warming up. Try again in 20 seconds."
            )

        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Invalid HF token. Check your HF_API_TOKEN."
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"HF Router API error: {response.status_code} - {response.text}"
            )

        # Parse response
        result = response.json()
        logger.info(f"Parsed result: {result}")

        # Router returns: [{"label": "POSITIVE", "score": 0.9987}, ...]
        # OR: [[{"label": "POSITIVE", "score": 0.9987}, ...]]
        if isinstance(result, list):
            if isinstance(result[0], list):
                predictions = result[0]
            else:
                predictions = result
        else:
            predictions = [result]

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
            detail="Cannot connect to HF Router. Please try again."
        )
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise HTTPException(status_code=504, detail="Request timed out.")
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
