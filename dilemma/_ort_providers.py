"""Device-agnostic onnxruntime execution-provider selection.

The same code should run CPU on a Mac and CUDA on an NVIDIA box with no edits,
so ONNX sessions ask for providers via `ort_providers()` instead of hardcoding
one. Default order: CUDA if the installed onnxruntime exposes it (i.e. a GPU +
onnxruntime-gpu), else CPU. CPU is always the final fallback.

Override with the env var (comma-separated, highest priority first), e.g.:
    DILEMMA_ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider   # force GPU
    DILEMMA_ORT_PROVIDERS=CoreMLExecutionProvider,CPUExecutionProvider # try Apple NE
    DILEMMA_ORT_PROVIDERS=CPUExecutionProvider                         # force CPU

CoreML is opt-in only (op-coverage for these models varies); the auto path never
picks it, so Mac behavior is unchanged unless explicitly requested.
"""
from __future__ import annotations

import os


def ort_providers() -> list[str]:
    """The execution providers to hand `onnxruntime.InferenceSession(providers=...)`,
    best-first. Honors DILEMMA_ORT_PROVIDERS; else auto-detects CUDA vs CPU."""
    override = os.environ.get("DILEMMA_ORT_PROVIDERS", "").strip()
    if override:
        return [p.strip() for p in override.split(",") if p.strip()]
    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - onnxruntime absent/broken: caller handles
        return ["CPUExecutionProvider"]
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:      # NVIDIA + onnxruntime-gpu
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" not in providers:   # always a fallback
        providers.append("CPUExecutionProvider")
    return providers
