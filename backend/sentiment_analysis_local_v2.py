"""
Sentiment Analysis Model Loader - Local Testing with Edge Cases
================================================================

This script demonstrates loading a pre-trained sentiment analysis model
(DistilBERT) and running inference on sample texts, including edge cases.

Key concepts:
- Model loading: Download once, cache forever
- Tokenization: Convert text to numbers
- Inference: Push tokens through neural network
- Post-processing: Convert logits to human-readable sentiment
- Edge case handling: How does the model handle weird inputs?

Usage:
    python sentiment_analysis_local_v2.py

First run: Downloads the model (~268 MB, takes 1-2 minutes)
Subsequent runs: Uses cached model (instant)
"""

import logging
import time
from typing import Dict, Any
from transformers import pipeline

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_sentiment_model() -> pipeline:
    """
    Load the DistilBERT sentiment analysis model.
    
    Why this approach?
    - "pipeline" abstracts away tokenization + inference + post-processing
    - First call downloads and caches the model
    - Subsequent calls use the cached model
    
    The model:
    - Name: "distilbert-base-uncased-finetuned-sst-2-english"
    - Size: 268 MB
    - Task: Sentiment classification (POSITIVE, NEGATIVE, NEUTRAL)
    - Accuracy: ~91% on standard benchmarks
    
    Returns:
        A Hugging Face pipeline object ready for inference
        
    Raises:
        Exception: If model download fails (network error, disk space)
    """
    logger.info("Loading sentiment analysis model...")
    
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # -1 = CPU, 0 = GPU (if available)
        )
        
        logger.info("✅ Model loaded successfully")
        return sentiment_pipeline
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        logger.error("Possible causes:")
        logger.error("  - No internet connection during first run")
        logger.error("  - Not enough disk space (~2GB required)")
        logger.error("  - Corrupted cache (try deleting ~/.cache/huggingface/)")
        raise


# ============================================================================
# INFERENCE
# ============================================================================
def analyze_sentiment(model: pipeline, text: str) -> Dict[str, Any]:
    """
    Run sentiment analysis on a single text.
    
    The process:
    1. Input: raw text (string)
    2. Tokenization: convert to numbers (handled by pipeline)
    3. Inference: push through neural network (handled by pipeline)
    4. Post-processing: convert raw outputs to human-readable labels
    5. Output: {"label": "POSITIVE", "score": 0.998}
    
    Args:
        model: Loaded sentiment pipeline
        text: Text to analyze (any length, but will be truncated to 512 tokens)
        
    Returns:
        Dictionary with:
        - "label": "POSITIVE", "NEGATIVE"
        - "score": confidence (0.0 to 1.0)
        - "inference_time_ms": how long inference took
        
    Note:
        This model only recognizes POSITIVE vs NEGATIVE.
        Neutral sentiment is handled by low confidence (e.g., 0.51 = barely positive).
    """
    try:
        # Time the inference to understand performance
        start_time = time.time()
        
        result = model(text, truncation=True, max_length=512)
        
        inference_time = time.time() - start_time
        
        # result is a list with one dict: [{"label": "POSITIVE", "score": 0.998}]
        sentiment = result[0]
        sentiment["inference_time_ms"] = round(inference_time * 1000, 2)
        
        return sentiment
        
    except Exception as e:
        logger.error(f"Inference failed for text: {text[:50]}...")
        logger.error(f"Error: {str(e)}")
        return {
            "label": "ERROR",
            "score": 0.0,
            "error": str(e)
        }


# ============================================================================
# TEST CASES
# ============================================================================
def run_test_cases(model: pipeline) -> None:
    """
    Test sentiment analysis on diverse examples.
    
    Why these test cases?
    - Positive: Straightforward positive sentiment
    - Negative: Straightforward negative sentiment
    - Neutral: Mixed/neutral sentiment (model struggles here)
    - Sarcasm: "Oh great, another delay" (model often gets this wrong)
    - Mixed: "I love the UI but hate the performance"
    """
    
    test_cases = [
        {
            "text": "I absolutely love this product! It's amazing!",
            "category": "POSITIVE",
            "explanation": "Clear positive language with exclamation marks"
        },
        {
            "text": "This is terrible and broken. Complete waste of money.",
            "category": "NEGATIVE",
            "explanation": "Strong negative words and tone"
        },
        {
            "text": "The weather is cloudy today.",
            "category": "NEUTRAL",
            "explanation": "Factual statement, no sentiment"
        },
        {
            "text": "Oh great, another bug. Just what I needed.",
            "category": "SARCASM (hard!)",
            "explanation": "Sarcastic negativity masked as positive words"
        },
        {
            "text": "I love the design but the performance is awful.",
            "category": "MIXED",
            "explanation": "Both positive and negative sentiment"
        }
    ]
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING CORE TEST CASES")
    logger.info("="*80)
    
    for i, test in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {test['category']}")
        logger.info(f"Text: {test['text']}")
        logger.info(f"Explanation: {test['explanation']}")
        
        result = analyze_sentiment(model, test['text'])
        
        logger.info(f"Result: {result['label']} ({result['score']:.1%} confidence)")
        logger.info(f"Inference time: {result.get('inference_time_ms', 'N/A')}ms")
        logger.info("-" * 80)


# ============================================================================
# EDGE CASE TESTING
# ============================================================================
def run_edge_case_tests(model: pipeline) -> None:
    """
    Test sentiment analysis on edge cases to understand limitations.
    
    Edge cases include:
    - Very short text (1-3 characters)
    - Empty or whitespace-only strings
    - Very long text (will be truncated)
    - Negations ("not bad", "not terrible")
    - Emojis
    - ALL CAPS
    - URLs
    - Random gibberish
    
    This helps you understand what the model handles well and what it struggles with.
    """
    
    edge_cases = [
        ("wow", "Very short (3 chars)"),
        ("I", "Single character"),
        ("a " * 300, "Very long (will truncate to 512 tokens)"),
        ("Not bad at all!", "Negation - tricky!"),
        ("This is not terrible", "Negation - double negative"),
        ("😊 great!", "Emoji (usually ignored)"),
        ("EXCELLENT!!!", "All caps"),
        ("https://example.com", "URL (tokenizer breaks it)"),
        ("test123 456 abc", "Random text/gibberish"),
        ("   ", "Only whitespace"),
        (".", "Only punctuation"),
        ("lol", "Internet slang"),
        ("This movie is okay", "Ambiguous (neither good nor bad)"),
        ("good good good", "Repeated word"),
        ("bad bad bad", "Repeated negative word"),
    ]
    
    logger.info("\n" + "="*80)
    logger.info("RUNNING EDGE CASE TESTS")
    logger.info("="*80)
    logger.info("\nThese tests show how the model handles unusual inputs:\n")
    
    # Print header
    print(f"\n{'Text':<35} | {'Result':<15} | {'Confidence':<12} | {'Notes'}")
    print("-" * 85)
    
    for text, explanation in edge_cases:
        # Truncate long text for display
        display_text = text[:30] + "..." if len(text) > 30 else text
        
        result = analyze_sentiment(model, text)
        
        # Format confidence as percentage
        confidence = f"{result['score']:.1%}" if result['score'] != 0.0 else "N/A"
        
        # Determine if result is uncertain (close to 50/50)
        is_uncertain = result['score'] < 0.60 and result['score'] > 0.40
        uncertain_note = "⚠️  Uncertain" if is_uncertain else ""
        
        print(f"{display_text:<35} | {result['label']:<15} | {confidence:<12} | {explanation} {uncertain_note}")
    
    logger.info("\n" + "="*80)
    logger.info("KEY OBSERVATIONS FROM EDGE CASES:")
    logger.info("="*80)
    logger.info("✓ Very short text: Model still makes predictions")
    logger.info("✓ Emojis: Usually ignored (no special handling)")
    logger.info("✓ URLs: Broken into tokens, usually neutral")
    logger.info("✓ Negations: Can struggle ('not bad' might be negative)")
    logger.info("✓ ALL CAPS: Treated same as lowercase")
    logger.info("✓ Whitespace-only: May cause warnings but handled")
    logger.info("✓ Gibberish: Random confidence")
    logger.info("✓ Ambiguous: Low confidence (close to 0.50)")
    logger.info("-" * 80)


# ============================================================================
# MAIN
# ============================================================================
def main():
    """
    Main execution flow.
    
    This demonstrates the complete pipeline:
    1. Load model (first time: download, subsequent: use cache)
    2. Run core test cases
    3. Run edge case tests
    4. Measure performance
    """
    
    logger.info("\n" + "="*80)
    logger.info("SENTIMENT ANALYSIS - LOCAL MODEL LOADING TEST")
    logger.info("="*80)
    logger.info(f"Model: DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)")
    logger.info(f"Task: Sentiment classification (POSITIVE / NEGATIVE)")
    logger.info(f"Size: 268 MB (will cache after first download)")
    
    # Load the model
    try:
        model = load_sentiment_model()
    except Exception as e:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Run core test cases
    run_test_cases(model)
    
    # Run edge case tests
    run_edge_case_tests(model)
    
    # Single detailed analysis
    logger.info("\n" + "="*80)
    logger.info("SINGLE DETAILED ANALYSIS")
    logger.info("="*80)
    
    test_text = "This is the best thing ever!"
    logger.info(f"Input: {test_text}")
    
    result = analyze_sentiment(model, test_text)
    
    logger.info(f"Output:")
    logger.info(f"  Label: {result['label']}")
    logger.info(f"  Confidence: {result['score']:.4f} ({result['score']*100:.2f}%)")
    logger.info(f"  Inference time: {result.get('inference_time_ms', 'N/A')}ms")
    
    logger.info("\n" + "="*80)
    logger.info("✅ TEST COMPLETE")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Review the edge case results above")
    logger.info("2. Notice which cases the model handles well")
    logger.info("3. Notice which cases are uncertain or wrong")
    logger.info("4. Then move to STEP 3: Wrap this in a FastAPI server")


if __name__ == "__main__":
    main()