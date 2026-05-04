import pytest
from openai import AsyncOpenAI

import ai


def test_client_base_url(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    ai._get_client.cache_clear()
    client = ai._get_client()
    assert str(client.base_url).rstrip("/") == ai.OPENROUTER_BASE_URL.rstrip("/")


def test_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    ai._get_client.cache_clear()
    with pytest.raises(ValueError, match="OPEN_ROUTER_API_KEY"):
        ai._get_client()
