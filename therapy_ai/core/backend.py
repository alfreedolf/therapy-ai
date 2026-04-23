"""
therapy_ai/core/backend.py

BackendFactory: single point of hardware detection.
The rest of the codebase NEVER imports mlx or torch directly.
They always go through this factory.
"""

from __future__ import annotations

import platform
import importlib
import logging
from enum import Enum, auto
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class HardwareBackend(Enum):
    MLX = auto()    # Apple Silicon via MLX
    CUDA = auto()   # NVIDIA GPU via PyTorch / Unsloth
    CPU = auto()    # Fallback — slow but always works


@dataclass(frozen=True)
class BackendInfo:
    backend: HardwareBackend
    device_name: str
    available_memory_gb: float
    supports_finetuning: bool

    def __str__(self) -> str:
        ft = "✓ fine-tuning" if self.supports_finetuning else "✗ fine-tuning"
        return (
            f"[{self.backend.name}] {self.device_name} "
            f"| {self.available_memory_gb:.1f} GB | {ft}"
        )


class BackendFactory:
    """
    Detects available hardware and returns the appropriate backend.

    Usage:
        info   = BackendFactory.detect()
        engine = BackendFactory.get_inference_engine(info)
    """

    _instance: BackendInfo | None = None  # singleton cache

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def detect(cls, force: HardwareBackend | None = None) -> BackendInfo:
        """Detect hardware once and cache the result."""
        if cls._instance is not None and force is None:
            return cls._instance

        if force is not None:
            cls._instance = cls._build_info(force)
        elif cls._is_apple_silicon():
            cls._instance = cls._build_info(HardwareBackend.MLX)
        elif cls._is_cuda_available():
            cls._instance = cls._build_info(HardwareBackend.CUDA)
        else:
            cls._instance = cls._build_info(HardwareBackend.CPU)

        logger.info("Backend detected: %s", cls._instance)
        return cls._instance

    @classmethod
    def get_inference_engine(cls, info: BackendInfo | None = None) -> Any:
        """
        Return the correct FastLanguageModel class for inference.
        mlx-tune and unsloth share an identical API surface intentionally.
        """
        backend = (info or cls.detect()).backend
        
        if backend == HardwareBackend.MLX:
            try:
                # Try mlx_tune first (Unsloth-like API)
                import mlx_tune
                from mlx_tune import FastLanguageModel
                logger.info("Using mlx_tune FastLanguageModel")
                return FastLanguageModel
            except ImportError:
                logger.warning("mlx_tune unavailable — using mlx_lm standard load")
                # Fallback to standard mlx_lm (your _MLXCompatWrapper handles it perfectly)
                from mlx_lm import load
                return _MLXCompatWrapper(load)

        elif backend == HardwareBackend.CUDA:
            from unsloth import FastLanguageModel
            return FastLanguageModel

        else:  # CPU fallback via transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
            return _CPUCompatWrapper()

    @classmethod
    def get_trainer_class(cls, info: BackendInfo | None = None) -> Any:
        """Return LoRA/GRPO trainer class appropriate for the backend."""
        backend = (info or cls.detect()).backend

        if backend == HardwareBackend.MLX:
            from mlx_tune import Trainer
            return Trainer

        elif backend == HardwareBackend.CUDA:
            from unsloth import FastLanguageModel  # noqa: F401
            from trl import GRPOTrainer, SFTTrainer
            return {"sft": SFTTrainer, "grpo": GRPOTrainer}

        else:
            raise RuntimeError("CPU backend does not support fine-tuning.")

    @classmethod
    def reset(cls) -> None:
        """Clear cached detection — useful in tests."""
        cls._instance = None

    # ── Private helpers ───────────────────────────────────────────────────────

    @classmethod
    def _is_apple_silicon(cls) -> bool:
        if platform.system() != "Darwin":
            return False
        if platform.machine() not in ("arm64", "aarch64"):
            return False
        return importlib.util.find_spec("mlx") is not None

    @classmethod
    def _is_cuda_available(cls) -> bool:
        spec = importlib.util.find_spec("torch")
        if spec is None:
            return False
        import torch
        return torch.cuda.is_available()

    @classmethod
    def _build_info(cls, backend: HardwareBackend) -> BackendInfo:
        if backend == HardwareBackend.MLX:
            return BackendInfo(
                backend=backend,
                device_name=cls._get_apple_chip_name(),
                available_memory_gb=cls._get_apple_memory_gb(),
                supports_finetuning=True,
            )
        elif backend == HardwareBackend.CUDA:
            import torch
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return BackendInfo(
                backend=backend,
                device_name=props.name,
                available_memory_gb=props.total_memory / (1024 ** 3),
                supports_finetuning=True,
            )
        else:
            import psutil
            return BackendInfo(
                backend=backend,
                device_name="CPU",
                available_memory_gb=psutil.virtual_memory().available / (1024 ** 3),
                supports_finetuning=False,
            )

    @staticmethod
    def _get_apple_chip_name() -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2
            )
            return result.stdout.strip() or "Apple Silicon"
        except Exception:
            return "Apple Silicon"

    @staticmethod
    def _get_apple_memory_gb() -> float:
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2
            )
            return int(result.stdout.strip()) / (1024 ** 3)
        except Exception:
            return 16.0  # safe default


# ── Compatibility wrappers ────────────────────────────────────────────────────

class _MLXCompatWrapper:
    """Thin wrapper around mlx_lm.load to mimic FastLanguageModel.from_pretrained API."""

    def __init__(self, load_fn: Any):
        self._load = load_fn

    def from_pretrained(self, model_name: str, **kwargs: Any) -> tuple[Any, Any]:
        model, tokenizer = self._load(model_name)
        return model, tokenizer


class _CPUCompatWrapper:
    """CPU fallback using raw Transformers."""

    def from_pretrained(self, model_name: str, **kwargs: Any) -> tuple[Any, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        return model, tokenizer
