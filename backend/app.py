"""
Sentiment Analysis FastAPI Server
Uses Hugging Face Free Inference API
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
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/distilbert-base-uncased-finetuned-sst-2-english/v1/text-classification"

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
    logger.info(f"Model: {MODEL_ID}")
    logger.info("✅ Server is ready to receive requests")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down...")

# ============================================================================
# APP
# ============================================================================
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment using Hugging Face",
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
        "model": MODEL_ID
    }

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="Server is running"
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
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )

        inference_time = time.time() - start_time
        logger.info(f"HF API status: {response.status_code}")
        logger.info(f"HF API response: {response.text[:300]}")

        # Handle model loading
        if response.status_code == 503:
            error_data = response.json()
            wait_time = error_data.get("estimated_time", 20)
            raise HTTPException(
                status_code=503,
                detail=f"Model is loading. Please wait {int(wait_time)} seconds and try again."
            )

        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Invalid HF token."
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"HF API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        logger.info(f"Result: {result}")

        # Handle different response formats
        # Format 1: [[{"label": "POSITIVE", "score": 0.99}]]
        # Format 2: [{"label": "POSITIVE", "score": 0.99}]
        if isinstance(result[0], list):
            predictions = result[0]
        else:
            predictions = result

        best = max(predictions, key=lambda x: x["score"])

        logger.info(f"✅ {best['label']} | {best['score']:.4f} | {inference_time*1000:.2f}ms")

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
