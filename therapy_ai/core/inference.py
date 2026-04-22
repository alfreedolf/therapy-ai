"""
therapy_ai/core/inference.py

InferenceEngine: streaming generation with prompt assembly.
Builds ChatML-formatted prompts and streams tokens back to caller.
Domain types (Message, GenerationResult) now live in domain.py.
"""

from __future__ import annotations

import logging
from typing import Generator, Optional

from therapy_ai.core.model_manager import ModelManager, LoadedModel
from therapy_ai.core.backend import HardwareBackend
from therapy_ai.domain import Message, GenerationResult

logger = logging.getLogger(__name__)

# Message and GenerationResult are imported from domain.py above.
# They are re-exported here for convenience so callers can do:
#   from inference import InferenceEngine, Message
__all__ = ["InferenceEngine", "Message", "GenerationResult"]


class InferenceEngine:
    """
    Stateless generation engine. Receives a list of Messages and
    yields token strings via a streaming generator.

    Usage:
        engine = InferenceEngine(manager)
        for token in engine.stream(messages):
            print(token, end="", flush=True)
    """

    def __init__(self, manager: ModelManager, system_prompt: Optional[str] = None):
        self.manager = manager
        self.system_prompt = system_prompt or self._default_system_prompt()

    # ── Public API ────────────────────────────────────────────────────────────

    def stream(
        self,
        messages: list[Message],
        override_system: Optional[str] = None,
    ) -> Generator[str, None, GenerationResult]:
        """
        Stream tokens one by one. Returns a GenerationResult via StopIteration.value.

        Usage pattern:
            gen = engine.stream(messages)
            try:
                while True:
                    token = next(gen)
                    print(token, end="")
            except StopIteration as e:
                result = e.value
        """
        loaded = self.manager.load()   # idempotent
        system = override_system or self.system_prompt
        prompt = self._build_prompt(messages, system, loaded)

        backend = loaded.backend
        if backend == HardwareBackend.MLX:
            yield from self._stream_mlx(prompt, loaded)
        elif backend == HardwareBackend.LLAMACPP:
            yield from self._stream_llamacpp(prompt, loaded)
        elif backend == HardwareBackend.CUDA:
            yield from self._stream_cuda(prompt, loaded)
        else:
            yield from self._stream_cpu(prompt, loaded)

    def generate(
        self,
        messages: list[Message],
        override_system: Optional[str] = None,
    ) -> GenerationResult:
        """Blocking (non-streaming) generation. Collects all tokens."""
        tokens: list[str] = []
        gen = self.stream(messages, override_system)
        try:
            while True:
                tokens.append(next(gen))
        except StopIteration as e:
            result: GenerationResult = e.value
            result.text = "".join(tokens)
            return result

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_prompt(
        self,
        messages: list[Message],
        system: str,
        loaded: LoadedModel,
    ) -> str:
        """
        Convert message list → tokenizer chat_template string.
        Falls back to manual ChatML if the tokenizer has no template.
        """
        tokenizer = loaded.tokenizer
        chat = [{"role": "system", "content": system}]
        chat += [m.to_dict() for m in messages]

        try:
            return tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Manual ChatML fallback
            lines = [f"<|im_start|>system\n{system}<|im_end|>"]
            for m in messages:
                lines.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
            lines.append("<|im_start|>assistant\n")
            return "\n".join(lines)

    # ── Backend-specific streaming ────────────────────────────────────────────

    def _stream_mlx(
        self,
        prompt: str,
        loaded: LoadedModel,
    ) -> Generator[str, None, GenerationResult]:
        """MLX streaming via mlx_lm.generate or mlx_tune."""
        from mlx_lm import stream_generate

        cfg = loaded.config
        collected: list[str] = []
        prompt_toks = len(loaded.tokenizer.encode(prompt))

        for response in stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt=prompt,
            max_tokens=cfg.max_new_tokens,
            temp=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
        ):
            token_text: str = response.text if hasattr(response, "text") else response
            collected.append(token_text)
            yield token_text

        full_text = "".join(collected)
        return GenerationResult(
            text=full_text,
            prompt_tokens=prompt_toks,
            completion_tokens=len(collected),
            stopped_by="eos",
        )

    def _stream_llamacpp(
        self,
        prompt: str,
        loaded: LoadedModel,
    ) -> Generator[str, None, GenerationResult]:
        """
        llama.cpp streaming via llama-cpp-python's create_completion(stream=True).
        Works identically on Mac (Metal), NVIDIA (CUDA), and CPU.
        Note: loaded.tokenizer is None for this backend.
        """
        cfg = loaded.config
        model = loaded.model  # is a llama_cpp.Llama instance

        collected: list[str] = []
        for chunk in model.create_completion(
            prompt,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repeat_penalty=cfg.repetition_penalty,
            stream=True,
        ):
            token_text: str = chunk["choices"][0].get("text", "")
            if token_text:
                collected.append(token_text)
                yield token_text

        full_text = "".join(collected)
        # llama.cpp doesn't expose prompt token count easily without an extra call;
        # we estimate it or return 0 — sufficient for logging purposes.
        return GenerationResult(
            text=full_text,
            prompt_tokens=0,
            completion_tokens=len(collected),
            stopped_by="eos",
        )

    def _stream_cuda(
        self,
        prompt: str,
        loaded: LoadedModel,
    ) -> Generator[str, None, GenerationResult]:
        """CUDA streaming via Transformers TextIteratorStreamer."""
        import torch
        from transformers import TextIteratorStreamer
        import threading

        cfg = loaded.config
        tokenizer = loaded.tokenizer
        model = loaded.model

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_toks = inputs["input_ids"].shape[-1]

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=True,
            streamer=streamer,
        )

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        collected: list[str] = []
        for token_text in streamer:
            collected.append(token_text)
            yield token_text

        thread.join()
        return GenerationResult(
            text="".join(collected),
            prompt_tokens=prompt_toks,
            completion_tokens=len(collected),
            stopped_by="eos",
        )

    def _stream_cpu(
        self,
        prompt: str,
        loaded: LoadedModel,
    ) -> Generator[str, None, GenerationResult]:
        """CPU fallback — no true streaming, simulates token-by-token."""
        import torch
        from transformers import TextIteratorStreamer
        import threading

        tokenizer = loaded.tokenizer
        model = loaded.model
        cfg = loaded.config

        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_toks = inputs["input_ids"].shape[-1]

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=True,
            streamer=streamer,
        )
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        collected: list[str] = []
        for token_text in streamer:
            collected.append(token_text)
            yield token_text

        thread.join()
        return GenerationResult(
            text="".join(collected),
            prompt_tokens=prompt_toks,
            completion_tokens=len(collected),
            stopped_by="eos",
        )

    # ── Default persona ───────────────────────────────────────────────────────

    @staticmethod
    def _default_system_prompt() -> str:
        try:
            from pathlib import Path
            p = Path("config/prompts/system_prompt.txt")
            if p.exists():
                return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass

        return (
            "You are a compassionate, non-judgmental AI therapist. "
            "You use evidence-based techniques from CBT, DBT, and motivational "
            "interviewing to support the user in exploring their emotional life, "
            "relationships, and personal growth. "
            "You never diagnose. You always encourage professional help when appropriate. "
            "You keep all information strictly confidential."
        )
