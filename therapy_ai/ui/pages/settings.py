"""
therapy_ai/ui/pages/settings.py

Model selection, LoRA adapter management, privacy controls, and data wipe.
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from therapy_ai.core.backend import BackendFactory
from therapy_ai.core.model_manager import ModelConfig


PRESET_MODELS = {
    "Qwen3 8B Q4 (recommended)":  "Qwen/Qwen3-8B-Instruct-GGUF",
    "Gemma 3 4B Q4 (lightweight)": "google/gemma-3-4b-it-GGUF",
    "Llama 3.2 3B Q4 (fast)":      "meta-llama/Llama-3.2-3B-Instruct-GGUF",
    "Custom (enter manually)":     "__custom__",
}


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
        selected_label = st.selectbox("Select model preset", list(PRESET_MODELS.keys()))
        model_id = PRESET_MODELS[selected_label]

        if model_id == "__custom__":
            model_id = st.text_input(
                "HuggingFace model ID or local path",
                placeholder="org/model-name",
            )

        quant = st.selectbox(
            "Quantisation",
            ["q4_k_m", "q8_0", "f16", "none"],
            help="q4_k_m is the best balance of quality and speed for M4/RTX4060.",
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
