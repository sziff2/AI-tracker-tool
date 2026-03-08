"""
Thin wrapper around the Anthropic SDK.
All LLM calls in the system go through this module.
"""

import json
import logging
from typing import Any

import anthropic

from configs.settings import settings

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def call_llm(prompt: str, *, max_tokens: int | None = None, temperature: float | None = None) -> str:
    """
    Send a single user-turn prompt and return the text response.
    """
    client = get_client()
    resp = client.messages.create(
        model=settings.llm_model,
        max_tokens=max_tokens or settings.llm_max_tokens,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    logger.debug("LLM response length: %d chars", len(text))
    return text


def call_llm_json(prompt: str, **kwargs) -> Any:
    """
    Call the LLM and parse the response as JSON.
    Strips markdown fences if present.
    """
    raw = call_llm(prompt, **kwargs)
    # Strip ```json ... ``` if the model wraps its output
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]  # drop first line
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON: %s — raw: %s", exc, raw[:500])
        raise
