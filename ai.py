import json
import logging
import os
from functools import lru_cache
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Prices per million tokens
MODEL_PRICES: dict[str, dict[str, float]] = {
    # Google models
    "google/gemini-2.0-flash-lite-001": {"input": 0.075, "output": 0.30},
    "google/gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "google/gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    # OpenAI models
    "openai/gpt-5": {"input": 1.25, "output": 10.00},
    "openai/gpt-5-mini": {"input": 0.25, "output": 2.00},
    "openai/gpt-5-nano": {"input": 0.05, "output": 0.40},
    # Feel free to add more models here
}

T = TypeVar("T", bound=BaseModel)


def _get_provider() -> str:
    """Get the active LLM provider from environment. Defaults to openrouter."""
    return os.environ.get("LLM_PROVIDER", "openrouter").lower()


@lru_cache
def _get_openrouter_client() -> AsyncOpenAI:
    """Get cached AsyncOpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPEN_ROUTER_API_KEY not found in environment")
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


@lru_cache
def _get_ollama_client() -> AsyncOpenAI:
    """Get cached AsyncOpenAI client configured for Ollama."""
    return AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def _get_client() -> AsyncOpenAI:
    """Get the appropriate client based on LLM_PROVIDER env var."""
    provider = _get_provider()
    if provider == "ollama":
        return _get_ollama_client()
    return _get_openrouter_client()


def _log_usage(response) -> None:
    """Log token usage and cost extrapolation for 1M queries."""
    usage = getattr(response, "usage", None)
    if usage is None:
        logger.warning("No usage data in response")
        return

    model = getattr(response, "model", "unknown")
    input_tokens = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)

    # Handle reasoning tokens (may be in output_tokens_details)
    reasoning_tokens = 0
    output_details = getattr(usage, "output_tokens_details", None)
    if output_details:
        reasoning_tokens = getattr(output_details, "reasoning_tokens", 0) or 0

    # Get prices (default to 0 if model unknown)
    prices = MODEL_PRICES.get(model, {"input": 0, "output": 0})
    input_price = prices["input"]
    output_price = prices["output"]  # Also used for reasoning

    # Calculate cost for this single query
    single_input_cost = (input_tokens / 1_000_000) * input_price
    single_output_cost = (output_tokens / 1_000_000) * output_price
    single_reasoning_cost = (reasoning_tokens / 1_000_000) * output_price
    single_total = single_input_cost + single_output_cost + single_reasoning_cost

    # Extrapolate to 1M queries
    million_cost = single_total * 1_000_000

    logger.info(
        f"Token usage for {model}: "
        f"input={input_tokens}, output={output_tokens}, reasoning={reasoning_tokens} | "
        f"This query: ${single_total:.6f} | "
        f"1M queries: ${million_cost:,.2f} | "
        f"10M queries: ${million_cost * 10:,.2f}"
    )


async def responses(
    model: str,
    input: str | list,
    text_format: type[T] | None = None,
    **kwargs,
) -> T | Any:
    """
    Call the configured LLM provider with automatic token usage logging.

    Routes to Ollama (chat.completions) or OpenRouter (responses API) based
    on the LLM_PROVIDER environment variable.

    @dev: The intention of this function is to be used as a wrapper around the OpenAI Responses API,
    so the developer can view token usage and cost extrapolation of each query. If this
    abstraction becomes cumbersome, you may remove it, but it is recommended to observe your token usage.
    """
    provider = _get_provider()
    client = _get_client()

    if provider == "ollama":
        return await _ollama_responses(client, model, input, text_format, **kwargs)
    return await _openrouter_responses(client, model, input, text_format, **kwargs)


async def _openrouter_responses(
    client: AsyncOpenAI,
    model: str,
    input: str | list,
    text_format: type[T] | None = None,
    **kwargs,
) -> T | Any:
    """Call OpenRouter via the OpenAI Responses API."""
    if text_format is not None:
        response = await client.responses.parse(
            model=model,
            input=input,
            text_format=text_format,
            **kwargs,
        )
        _log_usage(response)
        return response.output_parsed
    else:
        response = await client.responses.create(
            model=model,
            input=input,
            **kwargs,
        )
        _log_usage(response)
        return response


async def _ollama_responses(
    client: AsyncOpenAI,
    model: str,
    input: str | list,
    text_format: type[T] | None = None,
    **kwargs,
) -> T | Any:
    """Call Ollama via the OpenAI Chat Completions API."""
    # Normalise input to a messages list
    if isinstance(input, str):
        messages = [{"role": "user", "content": input}]
    else:
        messages = input

    if text_format is not None:
        schema = text_format.model_json_schema()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_schema", "json_schema": {"name": text_format.__name__, "schema": schema}},
            extra_body={"num_ctx": 32768},
            **kwargs,
        )
        _log_usage(response)
        content = response.choices[0].message.content
        return text_format.model_validate(json.loads(content))
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        _log_usage(response)
        return response
