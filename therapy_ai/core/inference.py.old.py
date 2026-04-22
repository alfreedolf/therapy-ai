# UPDATED: llama.cpp stream, imports domain.py

from therapy_ai.domain import GenerationResult

class Inference:
    def generate(self, prompt: str) -> GenerationResult:
        # stub
        return GenerationResult(text="stub response", confidence=0.8)