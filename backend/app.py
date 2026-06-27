"""
Sentiment Analysis FastAPI Server
===================================

A production-grade REST API for sentiment analysis using DistilBERT.

Features:
- Model loading on startup (efficient caching)
- Input validation (empty text, length limits)
- Error handling (with proper HTTP status codes)
- Logging (for production debugging)
- CORS support (for React frontend)
- Auto-generated interactive API documentation (/docs)
- Health check endpoint
- Performance metrics (inference time)

Usage:
    python app.py
    
Or with uvicorn:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Visit: http://localhost:8000/docs for interactive API documentation
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline

# ============================================================================
# LOGGING SETUP
# ============================================================================
# Configure logging for production debugging and monitoring

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================
# These will be populated on server startup

# The sentiment analysis model (loaded once on startup)
sentiment_model = None

# ============================================================================
# MODELS (Pydantic - for request/response validation)
# ============================================================================
# These define the structure of incoming requests and outgoing responses

class AnalyzeRequest(BaseModel):
    """
    Request body for sentiment analysis.
    
    Attributes:
        text: The text to analyze (required, non-empty)
        max_length: Maximum text length to process (default: 512 tokens)
    
    Example:
        {
            "text": "I absolutely love this product!"
        }
    """
    text: str = Field(
        ...,  # Required (no default value)
        min_length=1,  # At least 1 character
        max_length=10000,  # At most 10,000 characters
        description="Text to analyze for sentiment"
    )
    max_length: Optional[int] = Field(
        512,
        description="Maximum number of tokens to process (default: 512)"
    )


class AnalyzeResponse(BaseModel):
    """
    Response body for sentiment analysis.
    
    Attributes:
        text: The input text (for reference)
        sentiment: The predicted sentiment (POSITIVE or NEGATIVE)
        confidence: Confidence score (0.0 to 1.0)
        inference_time_ms: How long inference took
    
    Example:
        {
            "text": "I absolutely love this product!",
            "sentiment": "POSITIVE",
            "confidence": 0.9987,
            "inference_time_ms": 142.35
        }
    """
    text: str = Field(description="The input text that was analyzed")
    sentiment: str = Field(description="POSITIVE or NEGATIVE")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
    inference_time_ms: float = Field(description="Time taken for inference in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Server status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(description="Whether the sentiment model is loaded")
    message: str = Field(description="Status message")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(description="Error type")
    detail: str = Field(description="Error description")
    status_code: int = Field(description="HTTP status code")

# ============================================================================
# MODEL LOADING
# ============================================================================
# Load the model when server starts (not per request)

def load_sentiment_model():
    """
    Load the DistilBERT sentiment analysis model.
    
    Why load on startup?
    - Model loading takes ~5 seconds
    - Inference takes ~200ms
    - If we load per request: every request waits 5+ seconds
    - If we load on startup: first request waits 5 sec, rest are fast
    
    Returns:
        A Hugging Face pipeline object ready for inference
        
    Raises:
        Exception: If model download or loading fails
    """
    logger.info("Loading sentiment analysis model...")
    logger.info("Model: distilbert-base-uncased-finetuned-sst-2-english")
    
    try:
        start_time = time.time()
        
        # Load the pipeline (handles tokenization + inference + post-processing)
        # device=-1 means use CPU (no GPU required)
        model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # -1 = CPU, 0+ = GPU (if available)
        )
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error("Possible causes:")
        logger.error("  - No internet connection (first run needs to download)")
        logger.error("  - Not enough disk space (~2GB required)")
        logger.error("  - Corrupted cache (try: rm -rf ~/.cache/huggingface/)")
        raise


# ============================================================================
# FASTAPI APP WITH LIFECYCLE EVENTS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    
    This runs code on:
    - Server startup: yield before
    - Server shutdown: code after yield
    
    Why?
    - Load model once on startup (not per request)
    - Clean up resources on shutdown
    """
    # STARTUP: Load model before accepting requests
    logger.info("=" * 80)
    logger.info("SERVER STARTUP")
    logger.info("=" * 80)
    
    global sentiment_model
    sentiment_model = load_sentiment_model()
    
    logger.info("✅ Server is ready to receive requests")
    logger.info("Visit http://localhost:8000/docs for API documentation")
    logger.info("=" * 80)
    
    yield  # Server runs while this is suspended
    
    # SHUTDOWN: Clean up resources
    logger.info("=" * 80)
    logger.info("SERVER SHUTDOWN")
    logger.info("=" * 80)
    logger.info("Cleaning up resources...")
    
    # Clear model from memory
    sentiment_model = None
    logger.info("✅ Resources cleaned up")


# Create FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="Analyze sentiment of text using DistilBERT",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# CORS CONFIGURATION
# ============================================================================
# Why CORS?
# React (localhost:3000) and FastAPI (localhost:8000) are different origins
# Browser blocks cross-origin requests for security
# We need to explicitly allow React to call this API

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # Development: React on localhost
        "http://localhost:5173",      # Vite dev server
        "https://vercel.app",         # Production: React on Vercel
        "https://*.vercel.app",       # Any Vercel subdomain
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

logger.info("✅ CORS configured for React frontend")

# ============================================================================
# ROUTES / ENDPOINTS
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
    description="Check if the server and model are healthy"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with status and model status
        
    HTTP Status Codes:
        200: Server is healthy and model is loaded
        503: Server is unavailable (model not loaded)
    """
    if sentiment_model is None:
        logger.warning("Health check failed: Model not loaded")
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is initializing."
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="Server is running and model is loaded"
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
    summary="Analyze sentiment of text",
    description="Analyze the sentiment (positive/negative) of provided text",
    responses={
        200: {"description": "Sentiment analysis successful"},
        400: {"description": "Bad request (invalid input)"},
        422: {"description": "Invalid request format"},
        500: {"description": "Server error during inference"}
    }
)
async def analyze_sentiment(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze sentiment of the provided text.
    
    Args:
        request: AnalyzeRequest with 'text' field
        
    Returns:
        AnalyzeResponse with sentiment label and confidence
        
    HTTP Status Codes:
        200: Analysis successful
        400: Invalid input (empty text, too long, etc.)
        422: Invalid request format (Pydantic validation failed)
        500: Server error during model inference
        
    Example:
        Request:
            POST /analyze
            {
                "text": "I absolutely love this product!"
            }
            
        Response:
            {
                "text": "I absolutely love this product!",
                "sentiment": "POSITIVE",
                "confidence": 0.9987,
                "inference_time_ms": 142.35
            }
    """
    # Validate that model is loaded
    if sentiment_model is None:
        logger.error("Model not available for inference")
        raise HTTPException(
            status_code=500,
            detail="Sentiment model is not loaded. Server is initializing."
        )
    
    # Validate input (Pydantic does basic checks, we add custom ones)
    text = request.text.strip()
    
    if not text:
        logger.warning("Inference request with empty text after stripping")
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty or whitespace only"
        )
    
    if len(text) > 10000:
        logger.warning(f"Inference request with text longer than 10000 chars: {len(text)}")
        raise HTTPException(
            status_code=400,
            detail="Text exceeds maximum length of 10,000 characters"
        )
    
    # Log the request
    logger.info(f"Analyzing text: {text[:50]}..." if len(text) > 50 else f"Analyzing text: {text}")
    
    try:
        # Run inference (time it for performance monitoring)
        start_time = time.time()
        
        result = sentiment_model(
            text,
            truncation=True,
            max_length=request.max_length
        )
        
        inference_time = time.time() - start_time
        
        # result is a list: [{"label": "POSITIVE", "score": 0.9987}]
        sentiment_result = result[0]
        
        # Log successful inference
        logger.info(
            f"✅ Inference successful | "
            f"Sentiment: {sentiment_result['label']} | "
            f"Confidence: {sentiment_result['score']:.4f} | "
            f"Time: {inference_time*1000:.2f}ms"
        )
        
        # Return formatted response
        return AnalyzeResponse(
            text=text,
            sentiment=sentiment_result["label"],
            confidence=sentiment_result["score"],
            inference_time_ms=round(inference_time * 1000, 2)
        )
        
    except Exception as e:
        # Log the error
        logger.error(f"❌ Inference failed: {str(e)}")
        logger.error(f"Text: {text[:100]}...")
        
        # Return error response
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@app.get(
    "/",
    tags=["Info"],
    summary="API information",
    description="Get basic information about this API"
)
async def root():
    """
    Root endpoint - provides API information.
    
    Returns:
        Dictionary with API info and links
    """
    return {
        "name": "Sentiment Analysis API",
        "version": "1.0.0",
        "description": "Analyze sentiment of text using DistilBERT",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST)",
            "docs": "/docs (interactive)",
            "redoc": "/redoc (alternative)"
        },
        "example": {
            "method": "POST",
            "url": "http://localhost:8000/analyze",
            "body": {"text": "I love this product!"}
        }
    }

# ============================================================================
# STARTUP LOGGING
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Log startup details."""
    logger.info("\n" + "=" * 80)
    logger.info("SENTIMENT ANALYSIS API - STARTUP")
    logger.info("=" * 80)
    logger.info("Endpoints available:")
    logger.info("  - GET  /          (API info)")
    logger.info("  - GET  /health    (Health check)")
    logger.info("  - POST /analyze   (Sentiment analysis)")
    logger.info("  - GET  /docs      (Interactive documentation)")
    logger.info("  - GET  /redoc     (Alternative documentation)")
    logger.info("=" * 80 + "\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    # --reload: Restart when code changes (development only)
    # --host 0.0.0.0: Listen on all network interfaces
    # --port 8000: Listen on port 8000
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )