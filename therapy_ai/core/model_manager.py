"""
therapy_ai/core/model_manager.py

ModelManager: load, unload, hot-swap models from config.
Swapping from Qwen3 8B → Gemma 3 4B is a config change, not a code change.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import yaml

from therapy_ai.core.backend import BackendFactory, BackendInfo, HardwareBackend

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Validated model configuration, loaded from default_config.yaml."""
    model_id: str                          # HF repo id or local path
    quantization: str = "q4_k_m"          # q4_k_m | q8_0 | f16 | none
    max_seq_length: int = 4096
    dtype: Optional[str] = None           # None = auto
    load_in_4bit: bool = True             # QLoRA flag for CUDA
    # LoRA adapter path (None = base model only)
    adapter_path: Optional[str] = None
    # Inference sampling defaults
    temperature: float = 0.75
    top_p: float = 0.90
    repetition_penalty: float = 1.1
    max_new_tokens: int = 512

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        model_cfg = data.get("model", {})
        infer_cfg = data.get("inference", {})
        return cls(
            model_id=model_cfg["model_id"],
            quantization=model_cfg.get("quantization", "q4_k_m"),
            max_seq_length=model_cfg.get("max_seq_length", 4096),
            dtype=model_cfg.get("dtype"),
            load_in_4bit=model_cfg.get("load_in_4bit", True),
            adapter_path=model_cfg.get("adapter_path"),
            temperature=infer_cfg.get("temperature", 0.75),
            top_p=infer_cfg.get("top_p", 0.90),
            repetition_penalty=infer_cfg.get("repetition_penalty", 1.1),
            max_new_tokens=infer_cfg.get("max_new_tokens", 512),
        )


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    config: ModelConfig
    backend: HardwareBackend
    lora_attached: bool = False


class ModelManager:
    """
    Manages the lifecycle of a single loaded model.

    Usage:
        manager = ModelManager.from_config("config/default_config.yaml")
        loaded  = manager.load()
        manager.attach_lora("data/adapters/my_adapter")
        manager.unload()
    """

    def __init__(self, config: ModelConfig, backend_info: BackendInfo | None = None):
        self.config = config
        self.backend_info: BackendInfo = backend_info or BackendFactory.detect()
        self._loaded: LoadedModel | None = None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str | Path = "config/default_config.yaml") -> "ModelManager":
        config = ModelConfig.from_yaml(config_path)
        return cls(config)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> LoadedModel:
        """Load model + tokenizer into memory. Idempotent — safe to call multiple times."""
        if self._loaded is not None:
            logger.info("Model already loaded: %s", self.config.model_id)
            return self._loaded

        logger.info(
            "Loading %s on %s backend ...",
            self.config.model_id,
            self.backend_info.backend.name,
        )

        engine = BackendFactory.get_inference_engine(self.backend_info)
        model, tokenizer = self._load_by_backend(engine)

        self._loaded = LoadedModel(
            model=model,
            tokenizer=tokenizer,
            config=self.config,
            backend=self.backend_info.backend,
        )

        logger.info("Model loaded successfully.")

        # Attach LoRA adapter immediately if configured
        if self.config.adapter_path:
            self.attach_lora(self.config.adapter_path)

        return self._loaded

    def unload(self) -> None:
        """Release model from memory. Frees GPU/MPS VRAM."""
        if self._loaded is None:
            return
        backend = self._loaded.backend
        if backend == HardwareBackend.MLX:
            import mlx.core as mx
            del self._loaded.model
            mx.metal.clear_cache()
        elif backend == HardwareBackend.CUDA:
            import torch
            del self._loaded.model
            torch.cuda.empty_cache()
        self._loaded = None
        logger.info("Model unloaded and memory released.")

    def swap(self, new_config: ModelConfig) -> LoadedModel:
        """Hot-swap to a different model without restarting the app."""
        logger.info("Swapping model: %s → %s", self.config.model_id, new_config.model_id)
        self.unload()
        self.config = new_config
        return self.load()

    def attach_lora(self, adapter_path: str | Path) -> None:
        """Merge a trained LoRA adapter onto the loaded base model."""
        if self._loaded is None:
            raise RuntimeError("Model must be loaded before attaching LoRA.")

        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        backend = self._loaded.backend
        if backend == HardwareBackend.MLX:
            from mlx_lm.utils import load_adapters
            self._loaded.model = load_adapters(self._loaded.model, str(adapter_path))
        elif backend == HardwareBackend.CUDA:
            from peft import PeftModel
            self._loaded.model = PeftModel.from_pretrained(
                self._loaded.model, str(adapter_path)
            )

        self._loaded.lora_attached = True
        logger.info("LoRA adapter attached from: %s", adapter_path)

    # ── Getters ───────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded is not None

    @property
    def loaded(self) -> LoadedModel:
        if self._loaded is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self._loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_by_backend(self, engine: Any) -> tuple[Any, Any]:
        backend = self.backend_info.backend

        if backend == HardwareBackend.MLX:
            # mlx-tune / mlx_lm unified API
            return engine.from_pretrained(
                self.config.model_id,
                max_seq_length=self.config.max_seq_length,
            )

        elif backend == HardwareBackend.CUDA:
            # unsloth API  (mirrors mlx-tune intentionally)
            return engine.from_pretrained(
                self.config.model_id,
                max_seq_length=self.config.max_seq_length,
                dtype=self.config.dtype,
                load_in_4bit=self.config.load_in_4bit,
            )

        else:  # CPU wrapper
            return engine.from_pretrained(
                self.config.model_id,
                device_map="cpu",
            )
