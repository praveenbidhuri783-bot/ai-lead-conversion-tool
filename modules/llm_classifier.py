"""Optional LLM-based lead classification layer.

This module is entirely opt-in. Rule-based scoring (``modules.scorer``) is
always the fallback and the primary scoring path. When the user enables
"Use Advanced AI Analysis" in the UI and a supported API key is available in
the environment, this module asks an LLM to return a strict JSON structure
describing the lead, which is used to enrich (not replace) the rule-based
output.

Mobile numbers and customer names/identifiers are masked before any text is
sent to the LLM.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional

REQUIRED_KEYS = {
    "sentiment",
    "intent",
    "primary_objection",
    "secondary_objections",
    "conversion_probability",
    "reason",
    "recommended_action",
    "recommended_pitch",
}

_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\-\s]{7,13}\d)(?!\d)")

_SYSTEM_PROMPT = (
    "You are a strict JSON-only classifier for AI voice call summaries in a lead "
    "conversion scoring tool. Given a call summary, return ONLY a JSON object "
    "matching exactly this schema, with no extra commentary:\n"
    "{\n"
    '  "sentiment": "Positive|Neutral-Positive|Neutral|Neutral-Negative|Negative",\n'
    '  "intent": "High|Medium|Low|None",\n'
    '  "primary_objection": "string",\n'
    '  "secondary_objections": ["string", ...],\n'
    '  "conversion_probability": 0-100 integer,\n'
    '  "reason": "one sentence explanation",\n'
    '  "recommended_action": "one sentence next action for the agent",\n'
    '  "recommended_pitch": "one short agent pitch line"\n'
    "}"
)


class LLMClassificationError(Exception):
    """Raised when the LLM call fails or returns an unusable response."""


def mask_pii(text: str) -> str:
    """Mask phone numbers in free text before sending it to an external API."""
    if not text:
        return text
    return _PHONE_RE.sub(lambda m: "X" * (len(m.group(0)) - 4) + m.group(0)[-4:], text)


def is_llm_available() -> bool:
    """Whether any supported LLM API key is configured in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _call_anthropic(prompt_text: str, api_key: str, model: str) -> str:
    import anthropic  # imported lazily so the app runs without the package installed

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=400,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return "".join(block.text for block in response.content if hasattr(block, "text"))


def _call_openai(prompt_text: str, api_key: str, model: str) -> str:
    from openai import OpenAI  # imported lazily so the app runs without the package installed

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0,
        max_tokens=400,
    )
    return response.choices[0].message.content


def classify_with_llm(summary_text: str, provider: Optional[str] = None, model: Optional[str] = None) -> Dict:
    """Classify one call summary with an LLM, returning the parsed structured JSON.

    Raises ``LLMClassificationError`` on any failure so the caller can fall
    back to rule-based scoring without interrupting the batch.
    """
    masked_text = mask_pii(summary_text or "")
    if not masked_text.strip():
        raise LLMClassificationError("Empty call summary; nothing to classify.")

    provider = provider or ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai")
    prompt_text = f"Call summary:\n{masked_text}"

    try:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMClassificationError("ANTHROPIC_API_KEY is not set.")
            raw = _call_anthropic(prompt_text, api_key, model or "claude-sonnet-5")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise LLMClassificationError("OPENAI_API_KEY is not set.")
            raw = _call_openai(prompt_text, api_key, model or "gpt-4o-mini")
        else:
            raise LLMClassificationError(f"Unsupported LLM provider: {provider}")
    except LLMClassificationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMClassificationError(f"LLM call failed: {exc}") from exc

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw.lower().startswith("json") else raw

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMClassificationError(f"LLM returned invalid JSON: {exc}") from exc

    missing = REQUIRED_KEYS - set(parsed.keys())
    if missing:
        raise LLMClassificationError(f"LLM JSON is missing required keys: {missing}")

    return parsed
