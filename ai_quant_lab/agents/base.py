"""Anthropic SDK wrapper.

Centralizes:
    - retries with exponential backoff on transient errors
    - extracting the first JSON object from a response (Claude often wraps it
      in markdown fences or prose)
    - prompt caching for the system prompt when the same context is reused
      across multiple agents in a loop

Every other agent module talks to Claude exclusively through `call_claude`.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

from ai_quant_lab.config import settings


@dataclass(frozen=True)
class AgentMessage:
    role: str  # "user" or "assistant"
    content: str


@dataclass(frozen=True)
class AgentResponse:
    text: str
    usage: dict[str, int]
    model: str
    stop_reason: str | None = None


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def extract_first_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a Claude response.

    Tries direct json.loads first, then strips markdown fences, then matches
    the first {...} block. Raises ValueError if nothing parses.
    """
    cleaned = text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Found JSON-like block but could not parse: {exc}") from exc

    raise ValueError(f"No JSON object found in response:\n{cleaned[:500]}")


def call_claude(
    system: str,
    messages: Sequence[AgentMessage],
    *,
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.4,
    cache_system: bool = True,
    max_retries: int = 3,
    json_mode: bool = False,
) -> AgentResponse:
    """Dispatch to the configured LLM provider with optional fallback.

    Provider is `settings.llm_provider` (anthropic | groq | ollama).
    If the call fails and `settings.llm_fallback` is set, retries with the fallback.
    `json_mode=True` forces the model to emit syntactically valid JSON (Groq/Ollama
    support this natively; Anthropic ignores it since Claude rarely needs it).
    """
    primary = settings.llm_provider
    fallback = settings.llm_fallback
    kw = dict(model=model, max_tokens=max_tokens, temperature=temperature,
              cache_system=cache_system, max_retries=max_retries, json_mode=json_mode)
    try:
        return _dispatch(primary, system, messages, **kw)
    except (RuntimeError, ImportError) as exc:
        if not fallback or fallback == primary:
            raise
        return _dispatch(fallback, system, messages, **kw)


def _dispatch(provider: str, system, messages, **kw) -> AgentResponse:
    if provider == "anthropic":
        return _call_anthropic(system, messages, **kw)
    if provider == "groq":
        return _call_groq(system, messages, **kw)
    if provider == "ollama":
        return _call_ollama(system, messages, **kw)
    raise ValueError(f"Unknown llm_provider: {provider}")


def _call_anthropic(system, messages, *, model=None, max_tokens=2048, temperature=0.4,
                    cache_system=True, max_retries=3, json_mode=False) -> AgentResponse:
    api_key = settings.require_api_key()
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError("pip install anthropic") from exc

    client = Anthropic(api_key=api_key)
    use_model = model or settings.model
    system_blocks: list[dict[str, Any]] = (
        [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if cache_system else [{"type": "text", "text": system}]
    )
    api_messages = [{"role": m.role, "content": m.content} for m in messages]

    delay, last_error = 1.0, None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=use_model, max_tokens=max_tokens, temperature=temperature,
                system=system_blocks, messages=api_messages,
            )
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            return AgentResponse(
                text=text,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                    "cache_creation_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
                },
                model=response.model, stop_reason=response.stop_reason,
            )
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1: break
            time.sleep(delay); delay *= 2.0
    raise RuntimeError(f"Anthropic call failed after {max_retries} attempts: {last_error}")


def _call_groq(system, messages, *, model=None, max_tokens=2048, temperature=0.4,
               cache_system=True, max_retries=3, json_mode=False) -> AgentResponse:
    """Groq via OpenAI-compatible API. Free tier: ~1000 req/day on Llama 3.3 70B."""
    import httpx
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not set. Get one free at https://console.groq.com")
    use_model = model or settings.groq_model
    payload = {
        "model": use_model,
        "messages": [{"role": "system", "content": system}]
                    + [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {settings.groq_api_key}",
               "Content-Type": "application/json"}
    delay, last_error = 1.0, None
    for attempt in range(max_retries):
        try:
            r = httpx.post("https://api.groq.com/openai/v1/chat/completions",
                           json=payload, headers=headers, timeout=60.0)
            # 429 = rate limit. Honor Retry-After header if present, else longer backoff.
            if r.status_code == 429:
                retry_after = float(r.headers.get("retry-after", "0") or 0)
                wait = max(retry_after, 15.0 * (attempt + 1))
                last_error = RuntimeError(f"429 rate-limited (sleeping {wait:.0f}s)")
                if attempt == max_retries - 1: break
                time.sleep(wait); continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage_in = data.get("usage", {}).get("prompt_tokens", 0)
            usage_out = data.get("usage", {}).get("completion_tokens", 0)
            return AgentResponse(
                text=text,
                usage={"input_tokens": usage_in, "output_tokens": usage_out,
                       "cache_read_tokens": 0, "cache_creation_tokens": 0},
                model=use_model,
                stop_reason=data["choices"][0].get("finish_reason"),
            )
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1: break
            time.sleep(delay); delay *= 2.0
    raise RuntimeError(f"Groq call failed after {max_retries} attempts: {last_error}")


def _call_ollama(system, messages, *, model=None, max_tokens=2048, temperature=0.4,
                 cache_system=True, max_retries=3, json_mode=False) -> AgentResponse:
    """Ollama local. Requires `ollama pull <model>` first. No rate limits."""
    import httpx
    use_model = model or settings.ollama_model
    payload = {
        "model": use_model,
        "messages": [{"role": "system", "content": system}]
                    + [{"role": m.role, "content": m.content} for m in messages],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    delay, last_error = 1.0, None
    for attempt in range(max_retries):
        try:
            r = httpx.post(f"{settings.ollama_url}/api/chat",
                           json=payload, timeout=300.0)
            r.raise_for_status()
            data = r.json()
            text = data["message"]["content"]
            return AgentResponse(
                text=text,
                usage={"input_tokens": data.get("prompt_eval_count", 0),
                       "output_tokens": data.get("eval_count", 0),
                       "cache_read_tokens": 0, "cache_creation_tokens": 0},
                model=use_model,
                stop_reason=data.get("done_reason"),
            )
        except Exception as exc:
            last_error = exc
            if attempt == max_retries - 1: break
            time.sleep(delay); delay *= 2.0
    raise RuntimeError(
        f"Ollama call failed after {max_retries} attempts: {last_error}. "
        f"Is Ollama running? (ollama serve) Model pulled? (ollama pull {use_model})"
    )
