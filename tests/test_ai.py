import pytest
from unittest.mock import patch
from openai import AsyncOpenAI

import ai


def test_get_provider_defaults_to_openrouter(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert ai._get_provider() == "openrouter"


def test_get_provider_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert ai._get_provider() == "ollama"


def test_get_provider_openrouter_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert ai._get_provider() == "openrouter"


def test_ollama_client_base_url():
    # Clear lru_cache so env changes take effect
    ai._get_ollama_client.cache_clear()
    client = ai._get_ollama_client()
    assert str(client.base_url).rstrip("/") == ai.OLLAMA_BASE_URL.rstrip("/")


def test_openrouter_client_base_url(monkeypatch):
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    ai._get_openrouter_client.cache_clear()
    client = ai._get_openrouter_client()
    assert str(client.base_url).rstrip("/") == ai.OPENROUTER_BASE_URL.rstrip("/")


def test_get_client_returns_ollama_when_provider_ollama(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    ai._get_ollama_client.cache_clear()
    client = ai._get_client()
    assert str(client.base_url).rstrip("/") == ai.OLLAMA_BASE_URL.rstrip("/")


def test_get_client_returns_openrouter_when_provider_openrouter(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "test-key")
    ai._get_openrouter_client.cache_clear()
    client = ai._get_client()
    assert str(client.base_url).rstrip("/") == ai.OPENROUTER_BASE_URL.rstrip("/")
