# Therapy AI — Local Privacy-Preserving Psychotherapy System

A fully offline, encrypted AI-powered therapy assistant that runs on your local hardware.
No data ever leaves your device.

---

## Quick Start

### 1. Prerequisites
- Python 3.11+
- **Mac (Apple Silicon):** macOS 14+
- **NVIDIA:** CUDA 12.1+, cuDNN 8.9+

### 2. Clone and create environment

```bash
git clone <your-repo-url> therapy-ai
cd therapy-ai
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

**On MacBook Air M4 (recommended):**
```bash
pip install -e ".[mac,dev]"

# Install mlx-tune (Apple Silicon fine-tuning, not yet on PyPI)
pip install git+https://github.com/ARahim3/mlx-tune.git
```

**On NVIDIA GPU:**
```bash
pip install -e ".[cuda,dev]"
```

### 4. Verify hardware detection
```bash
python main.py info
```

### 5. Launch the UI
```bash
python main.py ui
# or directly:
streamlit run therapy_ai/ui/app.py
```

Open http://localhost:8501, set your passphrase (never stored), and start your first session.

---

## Project Structure

```
therapy-ai/
├── therapy_ai/
│   ├── core/
│   │   ├── backend.py          # Hardware detection (MLX / CUDA / CPU)
│   │   ├── model_manager.py    # Load, unload, hot-swap, LoRA attach
│   │   ├── inference.py        # Streaming generation engine
│   │   └── safety.py           # Crisis + injection screening
│   ├── memory/
│   │   ├── session_store.py    # AES-256-GCM encrypted SQLite
│   │   ├── context_builder.py  # Cross-session memory + truncation
│   │   └── exporter.py         # Sessions → JSONL for fine-tuning
│   ├── training/
│   │   ├── trainer.py          # LoRA SFT + GRPO (mlx-tune / unsloth)
│   │   ├── data_pipeline.py    # JSONL loading and formatting
│   │   └── reward.py           # GRPO reward function
│   └── ui/
│       ├── app.py              # Streamlit entry point
│       └── pages/
│           ├── chat.py         # Main therapy chat
│           ├── progress.py     # Mood charts + export
│           └── settings.py     # Model swap + privacy controls
├── config/
│   ├── default_config.yaml     # Model, inference, storage settings
│   ├── training_config.yaml    # LoRA / GRPO hyperparameters
│   └── prompts/
│       └── system_prompt.txt   # Therapy persona
├── scripts/
│   └── run_finetune.py         # CLI: export | train | preview
├── data/                       # .gitignored — all personal data here
├── .vscode/
│   ├── extensions.json
│   └── launch.json
├── pyproject.toml
└── main.py
```

---

## Switching Models

Edit `config/default_config.yaml`:
```yaml
model:
  model_id: "Qwen/Qwen3-8B-Instruct"   # change this line only
```
Or use the Settings page in the UI — no code changes needed.

### Recommended models by hardware

| Hardware         | Recommended model         | Quantisation | ~VRAM / RAM |
|------------------|---------------------------|--------------|-------------|
| M4 16 GB         | Qwen3 8B                  | q4_k_m       | ~5 GB       |
| M4 24 GB         | Qwen3 14B                 | q4_k_m       | ~9 GB       |
| RTX 4060 8 GB    | Gemma 3 4B                | q4_k_m       | ~3 GB       |
| RTX 4060 Ti 16 GB| Qwen3 8B                  | q4_k_m       | ~5 GB       |

---

## Fine-Tuning Workflow

### Step 1 — Accumulate sessions (50–100 recommended)
Use the app for daily sessions. Sessions auto-save encrypted.

### Step 2 — Export to JSONL
```bash
python scripts/run_finetune.py export --min-rating 4
```

### Step 3 — Run LoRA SFT
```bash
python scripts/run_finetune.py train --method sft
```

### Step 4 — Run GRPO (once you have ratings data)
```bash
python scripts/run_finetune.py train --method grpo
```

### Step 5 — Load your adapter
In the UI: Settings → LoRA Adapters → select your run → Load Adapter.

---

## Privacy Architecture

- **AES-256-GCM** encryption for all stored messages and session data
- **PBKDF2-SHA256** (480,000 iterations) key derivation from your passphrase
- **Zero telemetry**: Streamlit usage stats disabled, no external calls at runtime
- **Data sovereignty**: wipe everything from Settings → Privacy & Data
- **No passphrase storage**: key is derived in-memory on every app start

---

## Development

```bash
# Lint
ruff check .

# Format
ruff format .

# Type check
mypy therapy_ai/

# Tests
pytest
```

---

## Migrating to a New Hardware Platform

1. Copy your `data/` directory to the new machine.
2. Install the correct extras (`[mac]` or `[cuda]`).
3. `BackendFactory.detect()` auto-resolves the new backend.
4. No code changes required — the abstraction layer handles everything.
