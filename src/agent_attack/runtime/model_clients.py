from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from agent_attack.core.interfaces import VictimModel


@dataclass(slots=True)
class ClientConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    timeout_s: int = 60
    extra_headers: dict[str, str] = field(default_factory=dict)


class HTTPModelClient(VictimModel):
    """Unified client for vLLM(OpenAI-compatible), OpenAI, Gemini and Anthropic."""

    def __init__(self, config: ClientConfig) -> None:
        self.config = config
        self.provider = config.provider.lower()

    def respond(self, prompt: str) -> str:
        provider = self.provider
        if provider in {"vllm", "openai"}:
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
            }
            url = self._openai_like_url()
            headers = self._auth_headers("Bearer")
            data = self._post_json(url, payload, headers)
            return data["choices"][0]["message"]["content"]

        if provider == "gemini":
            key = self._require_key()
            model = self.config.model
            base = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
            url = f"{base}/models/{model}:generateContent?key={key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": self.config.temperature},
            }
            data = self._post_json(url, payload, {})
            return data["candidates"][0]["content"]["parts"][0]["text"]

        if provider == "anthropic":
            payload = {
                "model": self.config.model,
                "max_tokens": 512,
                "temperature": self.config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            base = self.config.base_url or "https://api.anthropic.com"
            url = f"{base.rstrip('/')}/v1/messages"
            headers = {
                "x-api-key": self._require_key(),
                "anthropic-version": "2023-06-01",
                **self.config.extra_headers,
            }
            data = self._post_json(url, payload, headers)
            blocks = data.get("content", [])
            for block in blocks:
                if block.get("type") == "text":
                    return block.get("text", "")
            return ""

        raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _openai_like_url(self) -> str:
        if self.provider == "openai":
            base = self.config.base_url or "https://api.openai.com/v1"
        else:
            if not self.config.base_url:
                raise ValueError("vLLM provider requires base_url, e.g. http://127.0.0.1:8000/v1")
            base = self.config.base_url
        return f"{base.rstrip('/')}/chat/completions"

    def _auth_headers(self, scheme: str) -> dict[str, str]:
        headers = {**self.config.extra_headers}
        api_key = self.config.api_key
        if api_key:
            headers["Authorization"] = f"{scheme} {api_key}"
        return headers

    def _require_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key
        env_map = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_key = env_map.get(self.provider)
        key = os.getenv(env_key) if env_key else None
        if not key:
            raise ValueError(f"Missing api_key for provider={self.provider}")
        return key

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        req_headers = {
            "Content-Type": "application/json",
            **headers,
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Model request failed: {exc.code} {detail}") from exc
        return json.loads(body)
