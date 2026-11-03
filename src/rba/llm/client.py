"""OpenAI-compatible chat completions. The API key is read from the environment, never from argv."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from rba.errors import RBAError


def chat(base_url: str, model: str, api_key_env: str, prompt: str, temperature: float, timeout_s: float) -> str:
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise RBAError(
            f"环境变量 {api_key_env} 没有设置。",
            "导出密钥后再跑，或加上 --dry-run，只把提示词写到磁盘。",
        )
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "你在读一份已经压缩过的模型证据。按用户给出的 JSON 结构回答，不要建议重新训练。",
            },
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RBAError(
            f"大模型接口返回 {exc.code}：{detail}",
            "检查 llm.base_url、llm.model 和密钥。分析缓存还在，修好之后不用 --fresh 也会接着用。",
        ) from exc
    except urllib.error.URLError as exc:
        raise RBAError(
            f"连不上大模型接口 {url}：{exc.reason}",
            "确认 base_url 是 OpenAI 兼容的 /v1 根地址，并且这台机器出得去网。",
        ) from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RBAError(
            "接口返回里没有 choices[0].message.content。",
            f"返回的键：{list(payload)[:8] if isinstance(payload, dict) else type(payload)}。确认这是 chat completions 协议。",
        ) from exc


def extract_json(text: str) -> dict | None:
    fence = "```"
    if fence in text:
        start = text.find(fence)
        rest = text[start + len(fence) :]
        if rest.startswith("json"):
            rest = rest[4:]
        end = rest.find(fence)
        if end != -1:
            candidate = rest[:end].strip()
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
