import { useState, useEffect, useRef, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAX_CHARS = 5000;
const MAX_HISTORY = 5;

// ──────────────────────────────────────────────
// Sub-components
// ──────────────────────────────────────────────

function Spinner() {
  return (
    <svg
      className="animate-spin h-5 w-5 text-white"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}

function ConfidenceBar({ value, color }) {
  // value is 0-100
  const [displayed, setDisplayed] = useState(0);

  // Animate the bar on mount
  useEffect(() => {
    const timer = setTimeout(() => setDisplayed(value), 50);
    return () => clearTimeout(timer);
  }, [value]);

  return (
    <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden" role="progressbar" aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}>
      <div
        className={`h-3 rounded-full progress-bar-fill ${color}`}
        style={{ width: `${displayed}%` }}
      />
    </div>
  );
}

function ResultCard({ result, onCopy, copied }) {
  const isPositive = result.sentiment === "POSITIVE";
  const confidencePct = (result.confidence * 100).toFixed(2);
  const inferenceMs = Math.round(result.inference_time_ms);

  // Confidence tier for color feedback
  const confidence = result.confidence;
  let barColor, tierLabel;
  if (confidence >= 0.85) {
    barColor = isPositive ? "bg-emerald-500" : "bg-red-500";
    tierLabel = "High confidence";
  } else if (confidence >= 0.6) {
    barColor = "bg-amber-400";
    tierLabel = "Medium confidence";
  } else {
    barColor = "bg-gray-400";
    tierLabel = "Low confidence";
  }

  return (
    <div className="animate-slide-up">
      {/* Sentiment Banner */}
      <div
        className={`rounded-2xl p-6 mb-4 text-center ${
          isPositive
            ? "bg-emerald-50 border-2 border-emerald-200"
            : "bg-red-50 border-2 border-red-200"
        }`}
      >
        <div className="text-4xl mb-2" aria-hidden="true">
          {isPositive ? "😊" : "😞"}
        </div>
        <p
          className={`text-3xl font-bold tracking-tight ${
            isPositive ? "text-emerald-600" : "text-red-500"
          }`}
        >
          {isPositive ? "Positive" : "Negative"}
        </p>
        <p className="text-sm text-gray-500 mt-1">Detected sentiment</p>
      </div>

      {/* Confidence Row */}
      <div className="bg-gray-50 rounded-2xl p-5 mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-600">Confidence</span>
          <span className="text-sm font-semibold text-gray-800">{confidencePct}%</span>
        </div>
        <ConfidenceBar value={parseFloat(confidencePct)} color={barColor} />
        <p className="text-xs text-gray-400 mt-2">{tierLabel}</p>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-gray-50 rounded-xl p-4 text-center">
          <p className="text-xs text-gray-500 mb-1">Inference time</p>
          <p className="text-lg font-semibold text-gray-800">{inferenceMs} ms</p>
        </div>
        <div className="bg-gray-50 rounded-xl p-4 text-center">
          <p className="text-xs text-gray-500 mb-1">Characters</p>
          <p className="text-lg font-semibold text-gray-800">{result.text.length}</p>
        </div>
      </div>

      {/* Copy Button */}
      <button
        onClick={onCopy}
        aria-label="Copy result to clipboard"
        className="w-full py-2.5 rounded-xl border border-gray-200 text-sm text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-all duration-150 flex items-center justify-center gap-2"
      >
        {copied ? (
          <>
            <span aria-hidden="true">✅</span> Copied!
          </>
        ) : (
          <>
            <span aria-hidden="true">📋</span> Copy result
          </>
        )}
      </button>
    </div>
  );
}

function HistoryItem({ item, index, onReuse }) {
  const isPositive = item.sentiment === "POSITIVE";
  const preview = item.text.length > 60 ? item.text.slice(0, 60) + "…" : item.text;

  return (
    <button
      onClick={() => onReuse(item.text)}
      aria-label={`Reuse text: ${preview}`}
      className="w-full text-left p-3 rounded-xl hover:bg-gray-50 border border-gray-100 hover:border-gray-200 transition-all duration-150 group"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs text-gray-600 leading-relaxed flex-1 line-clamp-2">{preview}</p>
        <span
          className={`shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full ${
            isPositive
              ? "bg-emerald-100 text-emerald-700"
              : "bg-red-100 text-red-600"
          }`}
        >
          {isPositive ? "+" : "−"}
        </span>
      </div>
      <p className="text-[10px] text-gray-400 mt-1 group-hover:text-gray-500 transition-colors">
        {(item.confidence * 100).toFixed(1)}% confidence · click to reuse
      </p>
    </button>
  );
}

// ──────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────

export default function SentimentAnalyzer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [history, setHistory] = useState([]);

  const textareaRef = useRef(null);
  const abortRef = useRef(null);

  // Auto-expand textarea height as user types
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 320)}px`;
  }, [text]);

  // Cancel in-flight request when component unmounts
  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const analyze = useCallback(async () => {
    // Inline validation so the closure always has the current `text`
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please enter some text before analyzing.");
      return;
    }
    if (trimmed.length < 3) {
      setError("Text is too short — please enter at least 3 characters.");
      return;
    }
    if (text.length > MAX_CHARS) {
      setError(`Text is too long — limit is ${MAX_CHARS.toLocaleString()} characters.`);
      return;
    }

    // Cancel any in-flight request
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    setResult(null);
    setCopied(false);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        // detail can be a string (400) or an array of objects (422 Pydantic)
        let msg;
        if (typeof data?.detail === "string") {
          msg = data.detail;
        } else if (Array.isArray(data?.detail) && data.detail.length > 0) {
          msg = data.detail.map((e) => e.msg ?? JSON.stringify(e)).join("; ");
        } else {
          msg = `Server returned ${response.status} ${response.statusText}`;
        }
        throw new Error(msg);
      }

      const data = await response.json();
      setResult(data);

      // Prepend to history, keep last MAX_HISTORY items
      setHistory((prev) => [data, ...prev].slice(0, MAX_HISTORY));
    } catch (err) {
      if (err.name === "AbortError") return; // user-triggered cancel, ignore
      if (err.message.includes("Failed to fetch") || err.message.includes("NetworkError")) {
        setError("Cannot reach the API — make sure the backend is running on " + API_URL);
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  }, [text]);

  // Submit on Ctrl+Enter or Cmd+Enter
  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      analyze();
    }
  };

  const handleCopy = async () => {
    if (!result) return;
    const payload = `Sentiment: ${result.sentiment}\nConfidence: ${(result.confidence * 100).toFixed(2)}%\nInference: ${Math.round(result.inference_time_ms)}ms\nText: ${result.text}`;
    try {
      await navigator.clipboard.writeText(payload);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setError("Clipboard access denied — please copy manually.");
    }
  };

  const handleReuseHistory = (reuseText) => {
    setText(reuseText);
    setResult(null);
    setError(null);
    textareaRef.current?.focus();
  };

  const handleClear = () => {
    setText("");
    setResult(null);
    setError(null);
    textareaRef.current?.focus();
  };

  const charCount = text.length;
  const charWarning = charCount > MAX_CHARS * 0.9;
  const overLimit = charCount > MAX_CHARS;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50 flex items-start justify-center py-12 px-4">
      <div className="w-full max-w-2xl">

        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-indigo-600 rounded-2xl mb-4 shadow-lg shadow-indigo-200">
            <span className="text-2xl" aria-hidden="true">🧠</span>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
            Sentiment Analyzer
          </h1>
          <p className="text-gray-500 mt-2 text-sm">
            Powered by DistilBERT · Enter any text to detect its sentiment
          </p>
        </div>

        {/* Main Card */}
        <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/60 border border-slate-100 p-6 md:p-8 mb-6">

          {/* Input Section */}
          <div className="mb-5">
            <label htmlFor="sentiment-input" className="block text-sm font-semibold text-gray-700 mb-2">
              Your text
            </label>

            <div className={`relative rounded-2xl border-2 transition-colors duration-150 ${
              overLimit
                ? "border-red-400 bg-red-50"
                : error && !result
                ? "border-red-300"
                : "border-gray-200 focus-within:border-indigo-400"
            }`}>
              <textarea
                id="sentiment-input"
                ref={textareaRef}
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  if (error) setError(null);
                }}
                onKeyDown={handleKeyDown}
                placeholder="Type or paste your text here… (⌘/Ctrl + Enter to analyze)"
                aria-label="Text to analyze"
                aria-describedby={error ? "input-error" : "char-count"}
                aria-invalid={!!error}
                disabled={loading}
                rows={4}
                className="w-full resize-none bg-transparent rounded-2xl px-4 pt-4 pb-10 text-gray-800 placeholder-gray-400 text-sm leading-relaxed focus:outline-none disabled:opacity-50"
                style={{ minHeight: "120px" }}
              />
              {/* Character count overlay */}
              <div
                id="char-count"
                className={`absolute bottom-3 right-4 text-xs ${
                  overLimit ? "text-red-500 font-semibold" : charWarning ? "text-amber-500" : "text-gray-400"
                }`}
                aria-live="polite"
              >
                {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()}
              </div>
            </div>

            {/* Inline error */}
            {error && (
              <p
                id="input-error"
                role="alert"
                className="mt-2 text-sm text-red-600 flex items-center gap-1.5 animate-fade-in"
              >
                <span aria-hidden="true">⚠️</span> {error}
              </p>
            )}
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={analyze}
              disabled={loading || overLimit}
              aria-label="Analyze sentiment"
              className="flex-1 py-3 px-6 bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white font-semibold rounded-xl transition-all duration-150 flex items-center justify-center gap-2 shadow-md shadow-indigo-200 hover:shadow-lg hover:shadow-indigo-200 disabled:shadow-none disabled:cursor-not-allowed text-sm"
            >
              {loading ? (
                <>
                  <Spinner />
                  Analyzing…
                </>
              ) : (
                <>
                  <span aria-hidden="true">🔍</span>
                  Analyze Sentiment
                </>
              )}
            </button>

            {(text || result) && (
              <button
                onClick={handleClear}
                disabled={loading}
                aria-label="Clear input and result"
                className="py-3 px-4 rounded-xl border-2 border-gray-200 text-gray-500 hover:border-gray-300 hover:text-gray-700 hover:bg-gray-50 transition-all duration-150 text-sm font-medium disabled:opacity-40"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        {/* Result Card */}
        {result && !loading && (
          <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/60 border border-slate-100 p-6 md:p-8 mb-6 animate-fade-in">
            <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-4">
              Analysis Result
            </h2>
            <ResultCard result={result} onCopy={handleCopy} copied={copied} />
          </div>
        )}

        {/* History */}
        {history.length > 0 && (
          <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/60 border border-slate-100 p-6 md:p-8 animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wider">
                Recent Analyses
              </h2>
              <button
                onClick={() => setHistory([])}
                className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
                aria-label="Clear history"
              >
                Clear all
              </button>
            </div>
            <div className="space-y-2">
              {history.map((item, i) => (
                <HistoryItem key={i} item={item} index={i} onReuse={handleReuseHistory} />
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <p className="text-center text-xs text-gray-400 mt-6">
          API endpoint: <code className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">{API_URL}/analyze</code>
        </p>
      </div>
    </div>
  );
}
