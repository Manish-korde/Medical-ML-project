# Model Workflow Guide

## Overview

This system uses **Groq API** with a single Llama model for all LLM tasks to ensure consistency, simplicity, and reliable deployment.

## LLM Model Selection

### Primary Model
- **Model:** `llama-3.3-70b-versatile`
- **Use Case:** All medical report generation tasks
- **Why:** Fast inference, reliable for medical summaries, cost-effective

### Fallback Behavior

If the Groq API is unavailable (network error, rate limit, invalid key), the system falls back to a **template-based response** that provides:
- Structured medical summary based on prediction results
- Basic recommendations derived from diagnosis/risk level

The fallback does NOT switch to a different model - it uses static templates to maintain consistency and avoid API complexity.

## Decision Logic

```
Groq API Available?
    ├─ Yes → Use llama-3.3-70b-versatile for reports
    └─ No  → Use template-based fallback (no model switching)
```

## When to Use This System

| Task | Model | Response Type |
|------|-------|-------------|
| Chest X-ray analysis | Llama 3.3 (Groq) | AI-generated report |
| Heart disease risk | Llama 3.3 (Groq) | AI-generated report |
| Complete analysis | Llama 3.3 (Groq) | AI-generated report |
| API unavailable | Template | Static summary |

## No Multiple Model Selection

This system intentionally uses **one model** for all tasks:
- No model switching based on task complexity
- No DeepSeek or other model integration
- No local Ollama dependency

This design ensures:
- Simpler deployment (no hardware dependencies)
- Predictable costs (API-only)
- Consistent output quality

## Modifying the Model

To change the Groq model, edit `src/api/predict.py`:

```python
# Line ~138
model="llama-3.3-70b-versatile",  # Change model name here
```

Any Groq-supported model can be used. Refer to [Groq API docs](https://console.groq.com/docs).