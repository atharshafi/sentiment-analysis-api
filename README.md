# Sentiment Analysis API 🎭

A production-grade **Sentiment Analysis API** built as an AI Engineering learning project. This project demonstrates how to build, deploy, and scale machine learning inference pipelines from local development to production.

> **Status:** STEP 1-2 Complete ✅ | STEP 3-5 In Progress 🚀

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Roadmap](#project-roadmap)
- [Learning Outcomes](#learning-outcomes)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## 🎯 Project Overview

This project teaches **AI Engineering** concepts by building a real, deployable application:

**What it does:**
- Takes user text as input (e.g., "I love this product!")
- Analyzes the sentiment using a pre-trained neural network
- Returns the sentiment classification (POSITIVE/NEGATIVE) with confidence scores

**Why this project?**
- Bridges web development (React, FastAPI) and AI/ML (pre-trained models)
- Covers the full stack: local development → API server → web frontend → cloud deployment
- Uses real production patterns (error handling, logging, caching, deployment)
- Small enough to complete in 2 weeks, but realistic enough for a portfolio

**Target audience:**
- Full-stack developers new to AI/ML
- People wanting to transition into AI engineering
- Anyone building the "hello world" of machine learning applications

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    User's Browser (Vercel)                  │
│                      React Frontend                          │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Input: "I love this product!"                       │   │
│  │  Submit Button → HTTP POST Request                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
                    HTTP POST (JSON)
                             │
┌────────────────────────────▼────────────────────────────────┐
│                  API Server (Render)                        │
│                   FastAPI Backend                           │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  POST /analyze                                       │   │
│  │  - Receive text                                      │   │
│  │  - Validate input                                    │   │
│  │  - Run inference (call model)                        │   │
│  │  - Return: {sentiment, confidence, time}            │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────┘
                             │
              DistilBERT Model (In Memory)
                             │
┌────────────────────────────▼────────────────────────────────┐
│              Pre-trained Neural Network                      │
│                   DistilBERT                                │
│                                                               │
│  Input: "I love this product!"                              │
│    ↓                                                         │
│  Tokenization: [I, love, this, product, !]                 │
│    ↓                                                         │
│  Embedding: [[0.2, 0.5], [0.8, 0.1], ...]                 │
│    ↓                                                         │
│  Neural Network Layers: [6 transformer blocks]              │
│    ↓                                                         │
│  Output Logits: [0.05, 5.23]                               │
│    ↓                                                         │
│  Softmax: [0.01%, 99.99%]                                  │
│    ↓                                                         │
│  Result: {"label": "POSITIVE", "score": 0.9999}            │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Step 1: Frontend captures text input
        ↓
Step 2: React sends POST request to API
        ↓
Step 3: FastAPI receives and validates
        ↓
Step 4: Load DistilBERT model (if not already loaded)
        ↓
Step 5: Tokenize text (words → numbers)
        ↓
Step 6: Run inference (push through neural network)
        ↓
Step 7: Post-process output (logits → probabilities → label)
        ↓
Step 8: Return JSON response to frontend
        ↓
Step 9: Frontend displays sentiment with visual feedback
```

---

## 💻 Tech Stack

### Backend (Python)

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.9+ | Programming language |
| **FastAPI** | 0.104+ | Web framework for REST API |
| **Uvicorn** | 0.24+ | ASGI server (runs FastAPI) |
| **transformers** | 4.36+ | Hugging Face library (model loading) |
| **torch** | 2.1+ | PyTorch (deep learning framework) |
| **python-dotenv** | 0.21+ | Environment variable management |

### Frontend (JavaScript/React)

| Tool | Version | Purpose |
|------|---------|---------|
| **React** | 18+ | UI framework |
| **Tailwind CSS** | 3+ | CSS framework (styling) |
| **Axios** or **Fetch API** | - | HTTP client for API calls |
| **Vercel** | - | Hosting platform |

### Infrastructure

| Tool | Purpose |
|------|---------|
| **GitHub** | Version control & deployment trigger |
| **Render** | Backend hosting (FastAPI server) |
| **Vercel** | Frontend hosting (React app) |
| **Docker** (optional) | Containerization |

---

## 📦 Prerequisites

Before starting, ensure you have:

- **Python 3.9 or 3.10** (check with `python --version`)
- **Node.js 16+ & npm** (check with `node --version` and `npm --version`)
- **Git** (check with `git --version`)
- **GitHub account** (for hosting code)
- **4GB RAM minimum** (for loading the model)
- **2GB disk space** (for model cache)
- **Internet connection** (for first-time model download)

### Quick Check

```bash
python --version       # Should show Python 3.9+
node --version        # Should show v16+
npm --version         # Should show 9+
git --version         # Should show 2.x
```

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
# Clone from GitHub
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-api.git
cd sentiment-analysis-api
```

Or create locally:

```bash
mkdir sentiment-analysis-api
cd sentiment-analysis-api
mkdir backend frontend
```

### Step 2: Backend Setup

```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# Verify activation (should show (venv) at terminal start)
which python    # or 'where python' on Windows

# Install dependencies
pip install --upgrade pip
pip install transformers torch python-dotenv

# Verify installation
python -c "import torch; import transformers; print('✅ All installed')"
```

**Expected output:**
```
✅ All installed
```

**Installation time:** 2-3 minutes

### Step 3: Test the Model Locally

```bash
# Make sure you're in backend folder with venv activated
(venv) ➜  backend python sentiment_analysis_local_v2.py
```

**First run (1-2 minutes):**
- Downloads DistilBERT model (~268 MB)
- Caches it locally
- Runs 5 core test cases
- Runs 15+ edge case tests

**Expected output:**
```
================================================================================
SENTIMENT ANALYSIS - LOCAL MODEL LOADING TEST
================================================================================

Loading sentiment analysis model...
✅ Model loaded successfully

================================================================================
RUNNING CORE TEST CASES
================================================================================

Test 1: POSITIVE
Text: I absolutely love this product! It's amazing!
Result: POSITIVE (99.98% confidence)
Inference time: 145.23ms
...

✅ TEST COMPLETE
```

**Second run (5 seconds):**
- Uses cached model
- Much faster (no download)

---

## 📖 Usage

### Current Status (STEP 1-2)

Right now, you can run sentiment analysis locally:

```bash
# Run the test script
(venv) ➜  backend python sentiment_analysis_local_v2.py
```

### What You Can Test

The script tests sentiment on:

**Core cases:**
- ✅ Positive: "I absolutely love this product!"
- ✅ Negative: "This is terrible and broken."
- ✅ Neutral: "The weather is cloudy today."
- ✅ Sarcasm: "Oh great, another bug."
- ✅ Mixed: "I love the UI but hate the performance."

**Edge cases:**
- ✅ Very short text ("wow", "I")
- ✅ Emojis ("😊 great!")
- ✅ URLs ("https://example.com")
- ✅ Negations ("Not bad at all!")
- ✅ ALL CAPS ("EXCELLENT!!!")
- ✅ Gibberish ("test123 456 abc")
- ✅ And 9 more...

### Example Output

```
Input: This is the best thing ever!
Output:
  Label: POSITIVE
  Confidence: 0.9987 (99.87%)
  Inference time: 142.35ms
```

---

## 🗺️ Project Roadmap

### STEP 1-2: ✅ COMPLETE

**Local Model Loading & Testing**

What was done:
- [x] Set up Python virtual environment
- [x] Installed transformers & torch
- [x] Loaded DistilBERT model locally
- [x] Created inference pipeline
- [x] Tested on 5 core cases
- [x] Tested on 15+ edge cases
- [x] Added logging & performance metrics
- [x] Pushed to GitHub

**Learnings:**
- How pre-trained models work
- Model downloading vs. loading
- Tokenization concept
- Inference performance (100-200ms on CPU)
- Caching benefits (first run slow, subsequent fast)
- Logging for production debugging

---

### STEP 3: 🚀 IN PROGRESS

**Build FastAPI REST API**

What will be done:
- [ ] Create FastAPI server
- [ ] Create `/analyze` endpoint
- [ ] Input validation (empty text, length limits)
- [ ] Error handling (model failures, timeouts)
- [ ] CORS configuration (allow requests from React)
- [ ] Logging & monitoring
- [ ] Local testing with curl

**Skills to learn:**
- REST API design
- HTTP methods (GET, POST)
- Request/response validation
- Error handling in production
- API testing & debugging

**Estimated time:** 1 day

---

### STEP 4: 📱 COMING SOON

**Create React Frontend UI**

What will be done:
- [ ] React component structure
- [ ] Text input field
- [ ] Submit button
- [ ] Loading spinner
- [ ] Display sentiment results
- [ ] Confidence visualizations
- [ ] Error handling & messages
- [ ] Responsive design (mobile + desktop)

**Skills to learn:**
- React hooks (useState, useEffect)
- API calls from frontend
- Component state management
- Async/await for HTTP requests
- Tailwind CSS styling
- UX best practices

**Estimated time:** 1.5 days

---

### STEP 5: 🌐 COMING SOON

**Deploy to Production**

What will be done:
- [ ] Deploy backend to Render
- [ ] Deploy frontend to Vercel
- [ ] Set up environment variables
- [ ] Configure CORS for production
- [ ] Test live application
- [ ] Monitor and debug

**Skills to learn:**
- Cloud deployment
- Environment variables
- CI/CD basics (GitHub → Render/Vercel)
- Cold starts & performance
- Production logging & monitoring

**Estimated time:** 1 day

**Final outcome:** Live app at `https://your-app.vercel.app` ✨

---

## 📚 Learning Outcomes

By completing this project, you'll understand:

### AI/ML Concepts
- ✅ What pre-trained models are
- ✅ How neural networks make predictions
- ✅ Model loading and inference
- ✅ Tokenization and embeddings
- ✅ Model caching and performance
- ✅ Handling edge cases

### Backend Engineering
- ✅ Building REST APIs with FastAPI
- ✅ Input validation and error handling
- ✅ Logging for production debugging
- ✅ Model serving best practices
- ✅ API design patterns

### Frontend Engineering
- ✅ React hooks and state management
- ✅ Async/await and promises
- ✅ HTTP requests from browsers
- ✅ Component composition
- ✅ Responsive UI design

### DevOps & Deployment
- ✅ Version control with Git
- ✅ CI/CD pipelines (GitHub → Cloud)
- ✅ Environment variables & secrets
- ✅ Cloud deployment platforms
- ✅ Monitoring & debugging

### Software Engineering
- ✅ Production-ready code patterns
- ✅ Error handling strategies
- ✅ Logging and observability
- ✅ Testing approaches
- ✅ Code documentation

---

## 🔧 Troubleshooting

### Issue: "No module named 'transformers'"

**Cause:** You haven't installed the library or aren't in the virtual environment.

**Solution:**
```bash
# Verify you're in venv (should see (venv) at terminal start)
source venv/bin/activate

# Install
pip install transformers torch
```

---

### Issue: "externally-managed-environment" error

**Cause:** You're not in the virtual environment.

**Solution:**
```bash
# Navigate to backend folder
cd sentiment-api/backend

# Activate virtual environment
source venv/bin/activate    # Mac/Linux
# OR
venv\Scripts\Activate.ps1   # Windows PowerShell

# Should see (venv) in terminal
# Then install
pip install transformers torch
```

---

### Issue: Model download fails / No internet

**Cause:** 
- No internet connection during first run
- Firewall blocking Hugging Face servers

**Solution:**
```bash
# Connect to internet
# Try again
python sentiment_analysis_local_v2.py

# If still fails, you can manually download later when online
```

---

### Issue: Slow inference (>500ms per prediction)

**Cause:** Running on CPU (normal for DistilBERT on CPU)

**Solution:**
- This is expected! 100-200ms is normal on CPU
- If you need faster: Get a GPU (NVIDIA with CUDA)
- Or use a lighter model (VADER, TextBlob)
- For this learning project, CPU is fine

---

### Issue: "CUDA out of memory"

**Cause:** You have a GPU but it's full or CUDA is enabled incorrectly.

**Solution:**
The code already uses CPU (`device=-1`), so this shouldn't happen.
If it does:
```bash
# Edit sentiment_analysis_local_v2.py
# Change: device=-1  (this means CPU)
# It should already be set correctly
```

---

### Issue: Model cache getting too large

**Cause:** Model files accumulate in cache (~268 MB per model)

**Solution:**
```bash
# See what's cached
ls -lh ~/.cache/huggingface/hub/

# Clear cache (will re-download on next run)
rm -rf ~/.cache/huggingface/

# Or on Windows:
rmdir C:\Users\<YourUsername>\.cache\huggingface /s
```

---

### Issue: Different results on second run

**Cause:** There shouldn't be any! Model weights are frozen.

**Solution:**
- This is a bug. Model outputs should be deterministic.
- Try clearing cache: `rm -rf ~/.cache/huggingface/`
- Restart Python/terminal
- Run again

---

## 📝 Project Structure

```
sentiment-analysis-api/
├── backend/                          # Python/FastAPI backend
│   ├── venv/                         # Virtual environment (don't commit)
│   ├── sentiment_analysis_local_v2.py # Local model testing
│   ├── requirements.txt              # Python dependencies
│   ├── app.py                        # FastAPI server (STEP 3)
│   ├── Procfile                      # Deployment config (STEP 5)
│   └── .env.example                  # Environment variables template
│
├── frontend/                         # React frontend
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── App.js                    # Main component (STEP 4)
│   │   └── index.js
│   ├── package.json
│   ├── .env.local                    # API endpoint
│   └── vercel.json                   # Vercel config (STEP 5)
│
├── .gitignore                        # Files to ignore in git
├── README.md                         # This file
└── docs/                             # Documentation
    ├── ARCHITECTURE.md
    ├── API_DOCS.md
    └── DEPLOYMENT.md
```

---

## 🎓 How to Learn from This Project

### Approach

This project is designed for **learning by building**, not just reading theory.

**Recommended learning style:**
1. **Read the concept explanation** (understand the why)
2. **Run the code** (see it in action)
3. **Experiment with the code** (modify and test)
4. **Debug when it breaks** (learn from failures)
5. **Explain to someone else** (test your understanding)

### Self-Check Questions

After each step, ask yourself:

**STEP 1-2 (Current):**
- [ ] What's the difference between a pre-trained model and training from scratch?
- [ ] Why do we download the model once and cache it?
- [ ] What is tokenization and why do models need it?
- [ ] What does "inference" mean?
- [ ] How does the model handle sarcasm?

**STEP 3 (Next):**
- [ ] What's a REST API and how is it different from a Python script?
- [ ] Why do we validate user input?
- [ ] What is CORS and why do we need it?

**STEP 4:**
- [ ] How does React communicate with the backend?
- [ ] What is state and why do we need it in React?
- [ ] How do we handle loading and error states?

**STEP 5:**
- [ ] What's the difference between localhost and production?
- [ ] Why do we use environment variables?
- [ ] How does GitHub trigger deployment?

---

## 🌟 Key Concepts Explained

### Model Loading vs. Inference

| Concept | What | When | Time | Result |
|---------|------|------|------|--------|
| **Download** | Get model file from internet | First run only | 1-2 min | Model saved to disk |
| **Load** | Put model in RAM | Every run | 5 sec | Model in memory, ready to use |
| **Inference** | Make predictions | Every request | 100-200ms | Sentiment label + confidence |

### Caching

**Without caching:**
```
Run 1: Download (1-2 min) + Load (5 sec) + Inference (200ms) = 2 min 5 sec
Run 2: Download (1-2 min) + Load (5 sec) + Inference (200ms) = 2 min 5 sec
Run 3: Download (1-2 min) + Load (5 sec) + Inference (200ms) = 2 min 5 sec
```

**With caching (our approach):**
```
Run 1: Download (1-2 min) + Load (5 sec) + Inference (200ms) = 2 min 5 sec  ← Slow but only once
Run 2: Load (5 sec) + Inference (200ms) = 5.2 sec                          ← Much faster!
Run 3: Load (5 sec) + Inference (200ms) = 5.2 sec                          ← Same speed
```

Caching saves 99% of time on subsequent runs!

### Confidence vs. Accuracy

| Term | Meaning | Example |
|------|---------|---------|
| **Confidence** | How sure the model is about its prediction | "99.87% confident it's POSITIVE" |
| **Accuracy** | How often the model is correct overall | "Model is 91% accurate on test data" |

A model can be **confident but wrong**. A confidence of 0.51 (51%) means the model is barely confident—it might actually be wrong.

---

## 📚 Additional Resources

### Understanding Transformers & Attention
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Academic paper)
- [Visual Transformer Explanation](https://jalammar.github.io/illustrated-transformer/) (Easy to understand)
- [3Blue1Brown on Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (Video series)

### FastAPI
- [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- [REST API Concepts](https://restfulapi.net/)

### React
- [React Official Docs](https://react.dev)
- [React Hooks Explained](https://react.dev/reference/react/hooks)

### Hugging Face
- [Hugging Face Hub](https://huggingface.co/models)
- [Transformers Library Docs](https://huggingface.co/docs/transformers/)

### Deployment
- [Render Docs](https://render.com/docs)
- [Vercel Docs](https://vercel.com/docs)

---

## 🤝 Contributing

This is a learning project, but you can fork and improve it!

Potential improvements:
- [ ] Add more sentiment models (VADER, RoBERTa, etc.)
- [ ] Add multi-language support
- [ ] Add sarcasm detection
- [ ] Add aspect-based sentiment analysis
- [ ] Add fine-tuning capability
- [ ] Add Docker containerization
- [ ] Add unit tests & integration tests
- [ ] Add API documentation (Swagger)

---

## 📄 License

This project is open source under the MIT License. See LICENSE file for details.

---

## 🙋 Questions & Support

### Having issues?

1. Check the **Troubleshooting** section above
2. Search GitHub issues: https://github.com/YOUR_USERNAME/sentiment-analysis-api/issues
3. Create a new issue with:
   - What you tried
   - What error you got
   - Your OS and Python version
   - Full error message/screenshot

### Want to contribute?

1. Fork the repository
2. Create a branch (`git checkout -b feature/your-feature`)
3. Make changes
4. Push branch (`git push origin feature/your-feature`)
5. Create Pull Request

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Estimated Time to Complete** | 2 weeks (part-time) |
| **Lines of Code (STEP 1-2)** | ~350 lines |
| **Model Size** | 268 MB |
| **Inference Speed** | 100-200ms (CPU) |
| **Accuracy** | ~91% on standard benchmarks |
| **Cloud Hosting Cost** | ~$0/month (free tiers) |

---

## 🎉 Completed Milestones

- [x] STEP 1-2: Local development setup & model loading
- [ ] STEP 3: FastAPI REST API
- [ ] STEP 4: React frontend
- [ ] STEP 5: Production deployment
- [ ] Portfolio-ready application ✨

---

## 🚀 Next Steps

**Ready to move forward?**

```bash
# Ask for STEP 3
Sentiment Analysis API - STEP 3: Build FastAPI Around the Model
```

You'll learn:
1. How REST APIs work
2. How to build with FastAPI
3. How to test APIs locally
4. How to prepare for deployment

---

**Last Updated:** 20th June 2026  
**Version:** 1.0 (STEP 1-2 Complete)  
**Maintainer:** Athar Shafi

---

> This README will be updated as each step is completed. Current progress: 40% (2/5 steps)
