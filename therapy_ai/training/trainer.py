"""
therapy_ai/training/trainer.py

Unified LoRA + GRPO trainer.
Uses mlx-tune on Apple Silicon and unsloth on NVIDIA — same API surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

import yaml

from therapy_ai.core.backend import BackendFactory, BackendInfo, HardwareBackend
from therapy_ai.core.model_manager import ModelManager, ModelConfig
from therapy_ai.training.data_pipeline import DataPipeline, DatasetStats

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 16                  # LoRA rank
    lora_alpha: int = 32         # scaling factor
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    use_rslora: bool = True      # Rank-Stabilized LoRA — better for small datasets


@dataclass
class TrainingConfig:
    method: str = "sft"              # "sft" | "grpo" | "dpo"
    output_dir: str = "data/adapters/run_001"
    num_epochs: int = 3
    batch_size: int = 2
    grad_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_seq_length: int = 2048
    save_steps: int = 50
    eval_steps: int = 50
    logging_steps: int = 10
    bf16: bool = True             # bfloat16 on both M-series and Ampere+
    seed: int = 42
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        tc = data.get("training", {})
        lora_data = tc.pop("lora", {})
        return cls(**tc, lora=LoRAConfig(**lora_data))


class LocalTrainer:
    """
    Unified training interface that dispatches to the correct backend.

    Usage:
        trainer = LocalTrainer.from_config(
            model_config_path="config/default_config.yaml",
            train_config_path="config/training_config.yaml",
        )
        trainer.train(
            train_data="data/finetune/train.jsonl",
            val_data="data/finetune/val.jsonl",
        )
    """

    def __init__(
        self,
        model_manager: ModelManager,
        train_config: TrainingConfig,
        backend_info: Optional[BackendInfo] = None,
    ):
        self.manager = model_manager
        self.train_config = train_config
        self.backend_info = backend_info or BackendFactory.detect()

    @classmethod
    def from_config(
        cls,
        model_config_path: str | Path = "config/default_config.yaml",
        train_config_path: str | Path = "config/training_config.yaml",
    ) -> "LocalTrainer":
        model_manager = ModelManager.from_config(model_config_path)
        train_config  = TrainingConfig.from_yaml(train_config_path)
        return cls(model_manager, train_config)

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        train_data: str | Path = "data/finetune/train.jsonl",
        val_data:   str | Path = "data/finetune/val.jsonl",
    ) -> Path:
        """
        Run training. Returns path to saved adapter.
        Dispatches to _train_mlx or _train_cuda depending on backend.
        """
        train_data = Path(train_data)
        val_data   = Path(val_data)

        if not train_data.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_data}\n"
                "Run: python scripts/run_finetune.py export first."
            )

        logger.info(
            "Starting %s fine-tuning on %s backend ...",
            self.train_config.method.upper(),
            self.backend_info.backend.name,
        )

        if self.backend_info.backend == HardwareBackend.MLX:
            return self._train_mlx(train_data, val_data)
        elif self.backend_info.backend == HardwareBackend.CUDA:
            return self._train_cuda(train_data, val_data)
        else:
            raise RuntimeError("CPU backend does not support fine-tuning.")

    # ── MLX (Apple Silicon) ───────────────────────────────────────────────────

    def _train_mlx(self, train_data: Path, val_data: Path) -> Path:
        """
        Fine-tune via mlx-tune. Supports SFT, DPO, and GRPO natively.
        https://github.com/ARahim3/mlx-tune
        """
        from mlx_tune import FastLanguageModel, Trainer as MlxTrainer

        tc = self.train_config
        output_path = Path(tc.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model, tokenizer = FastLanguageModel.from_pretrained(
            self.manager.config.model_id,
            max_seq_length=tc.max_seq_length,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=tc.lora.r,
            lora_alpha=tc.lora.lora_alpha,
            lora_dropout=tc.lora.lora_dropout,
            target_modules=tc.lora.target_modules,
            use_rslora=tc.lora.use_rslora,
        )

        trainer = MlxTrainer(
            model=model,
            tokenizer=tokenizer,
            method=tc.method,               # "sft" | "grpo" | "dpo"
            train_data=str(train_data),
            val_data=str(val_data),
            output_dir=str(output_path),
            num_epochs=tc.num_epochs,
            batch_size=tc.batch_size,
            learning_rate=tc.learning_rate,
            warmup_steps=tc.warmup_steps,
            max_seq_length=tc.max_seq_length,
            seed=tc.seed,
        )
        trainer.train()

        logger.info("MLX training complete. Adapter saved → %s", output_path)
        return output_path

    # ── CUDA (NVIDIA) ─────────────────────────────────────────────────────────

    def _train_cuda(self, train_data: Path, val_data: Path) -> Path:
        """
        Fine-tune via unsloth + TRL (SFT or GRPO).
        """
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset

        tc = self.train_config
        output_path = Path(tc.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model, tokenizer = FastLanguageModel.from_pretrained(
            self.manager.config.model_id,
            max_seq_length=tc.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=tc.lora.r,
            lora_alpha=tc.lora.lora_alpha,
            lora_dropout=tc.lora.lora_dropout,
            target_modules=tc.lora.target_modules,
            bias=tc.lora.bias,
            use_rslora=tc.lora.use_rslora,
        )

        # Use DataPipeline for validated, chat-template-formatted datasets
        pipeline = DataPipeline(train_data, val_data)
        stats = pipeline.inspect()
        logger.info("Pre-flight data check:\n%s", stats)

        if tc.method == "sft":
            train_ds, val_ds = pipeline.load_for_sft(
                tokenizer, max_seq_length=tc.max_seq_length
            )
            sft_config = SFTConfig(
                output_dir=str(output_path),
                num_train_epochs=tc.num_epochs,
                per_device_train_batch_size=tc.batch_size,
                gradient_accumulation_steps=tc.grad_accumulation_steps,
                learning_rate=tc.learning_rate,
                warmup_steps=tc.warmup_steps,
                save_steps=tc.save_steps,
                eval_steps=tc.eval_steps,
                logging_steps=tc.logging_steps,
                bf16=is_bfloat16_supported(),
                fp16=not is_bfloat16_supported(),
                seed=tc.seed,
                max_seq_length=tc.max_seq_length,
                report_to="none",      # no external telemetry
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                args=sft_config,
            )

        elif tc.method == "grpo":
            from trl import GRPOTrainer, GRPOConfig
            from reward import TherapyRewardFn

            grpo_train_ds = pipeline.load_for_grpo(
                tokenizer, max_seq_length=tc.max_seq_length
            )
            grpo_config = GRPOConfig(
                output_dir=str(output_path),
                num_train_epochs=tc.num_epochs,
                per_device_train_batch_size=tc.batch_size,
                learning_rate=tc.learning_rate,
                seed=tc.seed,
                bf16=is_bfloat16_supported(),
                report_to="none",
            )
            reward_fn = TherapyRewardFn()
            trainer = GRPOTrainer(
                model=model,
                tokenizer=tokenizer,
                reward_funcs=[reward_fn],
                args=grpo_config,
                train_dataset=grpo_train_ds,
            )
        else:
            raise ValueError(f"Unsupported training method: {tc.method}")

        trainer.train()
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        logger.info("CUDA training complete. Adapter saved → %s", output_path)
        return output_path
