"""
Piezo.AI — LLM Insight Generator
==================================
Generates AI-powered analysis of model performance and material predictions
using configured LLM provider (Google Gemini, OpenAI, Anthropic, or Ollama).

Called by the report builder when 'include_ai_insight' is True.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def generate_model_performance_insight(
    models: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    provider: str = "google",
    api_key: str = "",
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """
    Generate an AI analysis of model training performance.

    Returns a multi-paragraph text insight about:
    - Overall model quality assessment
    - Per-target performance analysis
    - Recommendations for improvement
    - Dataset quality observations

    Falls back to a rule-based summary if LLM call fails.
    """
    if not api_key or not api_key.strip():
        return _fallback_performance_insight(models, stats)

    prompt = _build_performance_prompt(models, stats)

    try:
        if provider.lower() in ("google", "gemini"):
            return _call_google(prompt, api_key, model_name, temperature, max_tokens)
        elif provider.lower() in ("openai", "openai-compatible"):
            return _call_openai(prompt, api_key, model_name, temperature, max_tokens)
        elif provider.lower() == "ollama":
            return _call_ollama(prompt, model_name, temperature, max_tokens)
        else:
            logger.warning(f"Unknown LLM provider: {provider}, using fallback")
            return _fallback_performance_insight(models, stats)
    except Exception as e:
        logger.error(f"LLM insight generation failed: {e}")
        return _fallback_performance_insight(models, stats)


def generate_material_prediction_insight(
    predictions: list[dict[str, Any]],
    *,
    provider: str = "google",
    api_key: str = "",
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.1,
    max_tokens: int = 2048,
) -> str:
    """
    Generate AI analysis of predicted material properties.

    Returns insights about potential applications, use-cases,
    and comparison with known materials based on d33/Tc/hardness values.
    """
    if not api_key or not api_key.strip():
        return _fallback_material_insight(predictions)

    prompt = _build_material_prompt(predictions)

    try:
        if provider.lower() in ("google", "gemini"):
            return _call_google(prompt, api_key, model_name, temperature, max_tokens)
        elif provider.lower() in ("openai", "openai-compatible"):
            return _call_openai(prompt, api_key, model_name, temperature, max_tokens)
        elif provider.lower() == "ollama":
            return _call_ollama(prompt, model_name, temperature, max_tokens)
        else:
            return _fallback_material_insight(predictions)
    except Exception as e:
        logger.error(f"LLM material insight failed: {e}")
        return _fallback_material_insight(predictions)


# ── Prompts ──────────────────────────────────────────────────

def _build_performance_prompt(models: list[dict], stats: dict) -> str:
    model_summary = []
    for m in models:
        model_summary.append(
            f"- {m.get('display_name', 'Unknown')}: target={m.get('target')}, "
            f"algorithm={m.get('algorithm')}, R2={m.get('r2_score', 0):.4f}, "
            f"RMSE={m.get('rmse', 0):.2f}, "
            f"train_samples={m.get('n_train_samples', 0)}, "
            f"test_samples={m.get('n_test_samples', 0)}"
        )

    return f"""You are an expert materials scientist and machine learning researcher specializing in piezoelectric ceramics.

Analyze the following ML model training results for a piezoelectric material discovery platform (Piezo.AI).

SYSTEM STATS:
- Total datasets: {stats.get('dataset_count', 0)}
- Total material rows: {stats.get('total_material_rows', 0)}
- Trained models: {stats.get('trained_model_count', 0)}
- Predictions made: {stats.get('prediction_count', 0)}

TRAINED MODELS:
{chr(10).join(model_summary)}

TARGET PROPERTY CONTEXT:
- d33 (pC/N): Piezoelectric charge coefficient. Higher = better piezoelectric response. Good ceramics: 100-600 pC/N.
- Tc (deg C): Curie temperature. Higher = better thermal stability. Good range: 200-450 deg C.
- Vickers Hardness (HV): Mechanical hardness. Higher = more durable. Good range: 200-600 HV.

Provide a concise, professional analysis (3-4 paragraphs) covering:
1. Overall assessment of model quality (R2 > 0.8 is good, > 0.9 is excellent, < 0.5 is poor)
2. Per-target performance evaluation and what the metrics mean for material discovery
3. Data quality observations (are there enough training samples?)
4. Specific recommendations to improve model accuracy

Keep the tone professional and scientific. Do not use markdown formatting, bullet points, or special characters. Write in plain paragraphs."""


def _build_material_prompt(predictions: list[dict]) -> str:
    pred_lines = []
    for p in predictions:
        parts = [f"Formula: {p.get('formula', 'Unknown')}"]
        if p.get("d33_predicted") is not None:
            parts.append(f"d33={p['d33_predicted']:.1f} pC/N")
        if p.get("tc_predicted") is not None:
            parts.append(f"Tc={p['tc_predicted']:.1f} deg C")
        if p.get("hardness_predicted") is not None:
            parts.append(f"HV={p['hardness_predicted']:.1f}")
        composite = "Composite" if p.get("is_composite") else "Bulk ceramic"
        parts.append(f"Type: {composite}")
        pred_lines.append(", ".join(parts))

    return f"""You are an expert materials scientist specializing in lead-free piezoelectric ceramics.

Analyze the following predicted piezoelectric material properties and provide insights about their potential applications.

PREDICTED MATERIALS:
{chr(10).join(pred_lines)}

PROPERTY RANGES FOR CONTEXT:
- d33: Sensors (30-100 pC/N), Actuators (200-400 pC/N), Transducers (100-300 pC/N), Energy harvesting (50-200 pC/N)
- Tc: High-temp applications need >300 deg C, room-temp devices need >150 deg C
- Hardness: Structural applications need >400 HV, standard devices 200-400 HV

Provide a concise analysis (2-3 paragraphs) covering:
1. Potential applications for each material based on its predicted properties
2. How these materials compare to well-known piezoelectric ceramics (PZT, BaTiO3, KNN)
3. Any notable observations about the composition-property relationships

Keep the tone professional and scientific. Do not use markdown formatting, bullet points, or special characters. Write in plain paragraphs."""


# ── LLM Provider Calls ─────────────────────────────────────

def _call_google(prompt: str, api_key: str, model: str, temp: float, max_tok: int) -> str:
    """Call Google Gemini API."""
    import urllib.request
    import urllib.error

    # Gemini REST API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temp,
            "maxOutputTokens": max_tok,
        },
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            candidates = body.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        logger.error(f"Gemini API error {e.code}: {error_body}")
        raise
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise

    return "AI insight generation returned empty response."


def _call_openai(prompt: str, api_key: str, model: str, temp: float, max_tok: int) -> str:
    """Call OpenAI-compatible API."""
    import urllib.request
    import urllib.error

    url = "https://api.openai.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": max_tok,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")

    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()

    return "AI insight generation returned empty response."


def _call_ollama(prompt: str, model: str, temp: float, max_tok: int) -> str:
    """Call local Ollama API."""
    import urllib.request

    url = "http://localhost:11434/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temp, "num_predict": max_tok},
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")

    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()


# ── Fallback (Rule-based) ──────────────────────────────────

def _fallback_performance_insight(models: list[dict], stats: dict) -> str:
    """Generate a rule-based performance summary when LLM is unavailable."""
    if not models:
        return "No trained models available for analysis."

    lines = []
    lines.append(
        f"The system has {stats.get('trained_model_count', len(models))} trained models "
        f"across {stats.get('dataset_count', 0)} datasets with "
        f"{stats.get('total_material_rows', 0)} total material entries."
    )

    good_models = [m for m in models if m.get("r2_score", 0) > 0.8]
    poor_models = [m for m in models if m.get("r2_score", 0) < 0.5]

    if good_models:
        names = ", ".join(m.get("display_name", "?") for m in good_models[:3])
        lines.append(
            f"Models with strong predictive performance (R2 > 0.8): {names}. "
            f"These models demonstrate reliable accuracy for material property prediction."
        )

    if poor_models:
        names = ", ".join(m.get("display_name", "?") for m in poor_models[:3])
        lines.append(
            f"Models requiring improvement (R2 < 0.5): {names}. "
            f"Consider increasing dataset size, applying feature selection, "
            f"or experimenting with alternative algorithms to improve accuracy."
        )

    low_data = [m for m in models if m.get("n_train_samples", 0) < 30]
    if low_data:
        lines.append(
            f"Note: {len(low_data)} model(s) were trained on fewer than 30 samples. "
            f"Increasing dataset size is strongly recommended for reliable predictions."
        )

    return " ".join(lines)


def _fallback_material_insight(predictions: list[dict]) -> str:
    """Generate rule-based material insight when LLM is unavailable."""
    if not predictions:
        return "No predictions selected for analysis."

    lines = []
    for p in predictions:
        formula = p.get("formula", "Unknown")
        parts = []
        d33 = p.get("d33_predicted")
        tc = p.get("tc_predicted")
        hv = p.get("hardness_predicted")

        if d33 is not None:
            if d33 > 200:
                parts.append("high piezoelectric response suitable for actuators")
            elif d33 > 100:
                parts.append("moderate piezoelectric response suitable for sensors")
            else:
                parts.append("low piezoelectric response")

        if tc is not None:
            if tc > 300:
                parts.append("excellent thermal stability for high-temperature applications")
            elif tc > 150:
                parts.append("adequate thermal stability for standard applications")

        if hv is not None:
            if hv > 400:
                parts.append("high mechanical durability")
            elif hv > 200:
                parts.append("moderate mechanical durability")

        if parts:
            lines.append(f"{formula}: {', '.join(parts)}.")

    if lines:
        return "Material Analysis: " + " ".join(lines)
    return "Predicted materials require further experimental validation."
