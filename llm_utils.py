"""
llm_utils.py
Generates a natural-language explanation for a forecast using retrieved
historical context (RAG).

SECURITY: API keys are read only from environment variables (loaded from a
local, git-ignored .env file via python-dotenv in app.py). No key is ever
hardcoded or logged in this file. If a key is missing or the call fails,
this module returns an error string instead of raising - a failed
explanation must never block the forecast itself from being shown.
"""

import os


def generate_explanation(forecast_value: float, target_date: str, pct_change: float, retrieved_docs: list):
    """
    Returns a (explanation_text, error_message) tuple. Exactly one of the
    two is None: on success explanation_text is set and error_message is
    None; on any failure explanation_text is None and error_message
    describes what went wrong.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    context = "\n".join(f"- {d['text']}" for d in retrieved_docs) or "No closely similar historical window was found."

    prompt = (
        "You are a forecasting analyst. Given:\n"
        f"- Forecast: {forecast_value:.2f} for {target_date} (change: {pct_change:+.1f}%)\n"
        f"- Similar historical patterns:\n{context}\n\n"
        "Write a concise (2-4 sentence) explanation of why the forecast shows "
        "this trend, referencing the historical patterns where relevant. "
        "Avoid speculation not grounded in the provided data."
    )

    try:
        if provider == "openai":
            return _call_openai(prompt), None
        return _call_gemini(prompt), None
    except Exception as exc:  # noqa: BLE001 - any provider/network failure surfaces to the UI, not a crash
        return None, f"LLM_ERROR: {exc}"


def _call_gemini(prompt: str) -> str:
    """Calls Gemini via Google's OpenAI-compatible endpoint, using the
    standard `openai` client pointed at Google's base URL. This sidesteps
    a known Google-side bug where newly issued "AQ."-prefix API keys are
    rejected by the native google-genai SDK / REST endpoint."""
    from openai import OpenAI

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your local .env file.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    response = client.chat.completions.create(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your local .env file.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()
