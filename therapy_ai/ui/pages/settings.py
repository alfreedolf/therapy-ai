"""
therapy_ai/ui/pages/settings.py

Model selection, LoRA adapter management, privacy controls, and data wipe.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from therapy_ai.core.backend import BackendFactory, HardwareBackend
from therapy_ai.core.model_manager import ModelConfig


# ── Backend-appropriate model presets ─────────────────────────────────────────
# Keys are display labels; values are HuggingFace model IDs.
# MLX models only run on Apple Silicon; CUDA/CPU use standard HF repos.

_PRESETS_MLX = {
    "Llama 3.1 8B 4-bit · MLX (recommended for M4 16 GB)": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "Qwen2.5 7B 4-bit · MLX (fast, M4 8 GB+)":             "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "Qwen2.5 14B 4-bit · MLX (M4 24 GB+)":                 "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "Custom (enter manually)":                              "__custom__",
}

_PRESETS_CUDA = {
    "Qwen2.5 7B · CUDA (RTX 4060 8 GB+, recommended)":     "Qwen/Qwen2.5-7B-Instruct",
    "Llama 3.1 8B · CUDA (RTX 4060 Ti 16 GB+)":            "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Gemma 2 2B · CUDA (low VRAM, <6 GB)":                 "google/gemma-2-2b-it",
    "Custom (enter manually)":                              "__custom__",
}

_PRESETS_CPU = {
    "Qwen2.5 7B · CPU (slow, no GPU required)":            "Qwen/Qwen2.5-7B-Instruct",
    "Gemma 2 2B · CPU (lighter, no GPU required)":         "google/gemma-2-2b-it",
    "Custom (enter manually)":                             "__custom__",
}


def _presets_for_backend(backend: HardwareBackend) -> dict[str, str]:
    if backend == HardwareBackend.MLX:
        return _PRESETS_MLX
    elif backend == HardwareBackend.CUDA:
        return _PRESETS_CUDA
    else:
        return _PRESETS_CPU


def render() -> None:
    st.markdown("## ⚙️ Settings")

    manager = st.session_state.manager
    store   = st.session_state.store

    # ── Hardware info ─────────────────────────────────────────────────────────
    with st.expander("🖥️ Hardware", expanded=True):
        info = BackendFactory.detect()
        st.markdown(f"**Backend:** `{info.backend.name}`")
        st.markdown(f"**Device:** {info.device_name}")
        st.markdown(f"**Available memory:** {info.available_memory_gb:.1f} GB")
        st.markdown(f"**Fine-tuning supported:** {'✅' if info.supports_finetuning else '❌'}")

    # ── Model selection ───────────────────────────────────────────────────────
    with st.expander("🤖 Model"):
        info = BackendFactory.detect()
        presets = _presets_for_backend(info.backend)

        selected_label = st.selectbox("Select model preset", list(presets.keys()))
        model_id = presets[selected_label]

        if model_id == "__custom__":
            model_id = st.text_input(
                "HuggingFace model ID or local path",
                placeholder="org/model-name",
            )

        quant_options = (
            ["4bit", "8bit", "none"]
            if info.backend == HardwareBackend.MLX
            else ["q4_k_m", "q8_0", "f16", "none"]
        )
        quant = st.selectbox(
            "Quantisation",
            quant_options,
            help=(
                "4bit = best quality/speed on Apple Silicon. "
                "q4_k_m = best for CUDA/CPU via llama.cpp."
            ),
        )

        if st.button("Apply Model", type="primary", disabled=not model_id):
            new_cfg = ModelConfig(model_id=model_id, quantization=quant)
            with st.spinner("Swapping model..."):
                manager.swap(new_cfg)
            st.session_state.manager = manager
            st.success(f"Switched to `{model_id}`")

    # ── LoRA adapters ─────────────────────────────────────────────────────────
    with st.expander("🧩 LoRA Adapters"):
        adapter_dir = Path("data/adapters")
        adapters = sorted(adapter_dir.glob("*")) if adapter_dir.exists() else []

        if adapters:
            selected_adapter = st.selectbox(
                "Available adapters",
                ["None"] + [a.name for a in adapters],
            )
            if st.button("Load Adapter"):
                if selected_adapter == "None":
                    st.info("No adapter loaded — using base model.")
                else:
                    with st.spinner("Attaching LoRA adapter..."):
                        manager.load()
                        manager.attach_lora(adapter_dir / selected_adapter)
                    st.success(f"Adapter `{selected_adapter}` attached ✅")
        else:
            st.info(
                "No adapters found in `data/adapters/`. "
                "Run a fine-tuning session first."
            )

    # ── System prompt ─────────────────────────────────────────────────────────
    with st.expander("📝 Therapy Persona"):
        prompt_path = Path("config/prompts/system_prompt.txt")
        current = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        new_prompt = st.text_area("System prompt", value=current, height=200)
        if st.button("Save Prompt"):
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(new_prompt, encoding="utf-8")
            st.session_state.engine.system_prompt = new_prompt
            st.success("System prompt updated.")

    # ── Privacy & data controls ───────────────────────────────────────────────
    with st.expander("🔒 Privacy & Data"):
        st.warning(
            "⚠️ The options below are **irreversible**. "
            "Your data cannot be recovered after deletion."
        )
        st.markdown("**Wipe ALL data** — deletes every session, message, and the encryption salt.")
        confirm = st.text_input(
            "Type DELETE to confirm permanent data wipe",
            placeholder="DELETE",
        )
        if st.button("Wipe All Data", type="primary", disabled=confirm != "DELETE"):
            store.wipe_all_data()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.success("All data permanently erased.")
            st.rerun()
