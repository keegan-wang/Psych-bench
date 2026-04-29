"""Model backend protocol + concrete implementations (lazy-imported)."""
from __future__ import annotations

import os
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    model: str

    def generate(self, prompt: str, stateful: bool = False) -> str: ...
    def reset(self) -> None: ...


class EchoBackend:
    """Deterministic offline backend for tests.

    Extracts the last standalone A/B/C letter in the prompt and returns it.
    Lets the test suite run without network or API keys.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.history: list[tuple[str, str]] = []

    def generate(self, prompt: str, stateful: bool = False) -> str:
        matches = re.findall(r"\b([ABC])\b", prompt)
        answer = matches[-1] if matches else "A"
        if stateful:
            self.history.append((prompt, answer))
        return answer

    def reset(self) -> None:
        self.history = []


class OpenAIBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self._conversation: list[dict] = []
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAI backend requires `pip install openai`"
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var not set")
        self._client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if stateful:
            self._conversation.append({"role": "user", "content": prompt})
            messages = list(self._conversation)
        else:
            messages = [{"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        if stateful:
            self._conversation.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._conversation = []


class AnthropicBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self._conversation: list[dict] = []
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Anthropic backend requires `pip install anthropic`"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY env var not set")
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, stateful: bool = False) -> str:
        if stateful:
            self._conversation.append({"role": "user", "content": prompt})
            messages = list(self._conversation)
        else:
            messages = [{"role": "user", "content": prompt}]
        resp = self._client.messages.create(
            model=self.model, max_tokens=256, messages=messages,
            temperature=0.0,
        )
        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", "") == "text"
        )
        if stateful:
            self._conversation.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._conversation = []


class HuggingFaceBackend:
    def __init__(self, model: str) -> None:
        self.model = model
        self._history: list[dict] = []
        try:
            from transformers import (  # type: ignore
                AutoModelForCausalLM, AutoTokenizer,
            )
            import torch  # type: ignore
        except ImportError as e:
            raise ImportError(
                "HuggingFace backend requires `pip install transformers torch`"
            ) from e
        token = os.environ.get("HF_TOKEN")
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, token=token, torch_dtype="auto", device_map="auto",
        )

    def generate(self, prompt: str, stateful: bool = False) -> str:
        torch = self._torch
        if stateful:
            self._history.append({"role": "user", "content": prompt})
            messages = list(self._history)
        else:
            messages = [{"role": "user", "content": prompt}]
        try:
            text_in = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text_in = prompt
        inputs = self._tokenizer(text_in, return_tensors="pt").to(
            self._model.device
        )
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(gen, skip_special_tokens=True).strip()
        if stateful:
            self._history.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._history = []


_BACKENDS = {
    "echo": EchoBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "huggingface": HuggingFaceBackend,
}


def get_backend(kind: str, model: str) -> ModelBackend:
    if kind not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{kind}'. Known: {list(_BACKENDS)}"
        )
    return _BACKENDS[kind](model)
