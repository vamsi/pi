"""Model configuration matching. Reads models.json to find the best vLLM config for a given model and hardware."""

from __future__ import annotations

import json
from pathlib import Path

from pi.pods.types import GPU

_MODELS_JSON = Path(__file__).parent / "models.json"
_models_data: dict | None = None


def _load_models() -> dict:
    global _models_data
    if _models_data is None:
        _models_data = json.loads(_MODELS_JSON.read_text())
    return _models_data


def get_model_config(
    model_id: str,
    gpus: list[GPU],
    requested_gpu_count: int,
) -> dict | None:
    """Get the best configuration for a model based on available GPUs.

    Returns a dict with keys: args, env (optional), notes (optional), or None.
    """
    models = _load_models()["models"]
    model_info = models.get(model_id)
    if not model_info:
        return None

    # Extract GPU type from the first GPU name (e.g., "NVIDIA H200" -> "H200")
    gpu_type = ""
    if gpus and gpus[0].name:
        gpu_type = gpus[0].name.replace("NVIDIA", "").strip().split(" ")[0]

    best_config = None

    for config in model_info.get("configs", []):
        if config.get("gpuCount", 1) != requested_gpu_count:
            continue

        gpu_types = config.get("gpuTypes", [])
        if gpu_types:
            type_matches = any(
                gpu_type in t or t in gpu_type for t in gpu_types
            )
            if not type_matches:
                continue

        best_config = config
        break

    # Fallback: match just GPU count without type check
    if not best_config:
        for config in model_info.get("configs", []):
            if config.get("gpuCount", 1) == requested_gpu_count:
                best_config = config
                break

    if not best_config:
        return None

    return {
        "args": list(best_config.get("args", [])),
        "env": dict(best_config["env"]) if best_config.get("env") else None,
        "notes": best_config.get("notes") or model_info.get("notes"),
    }


def is_known_model(model_id: str) -> bool:
    return model_id in _load_models()["models"]


def get_known_models() -> list[str]:
    return list(_load_models()["models"].keys())


def get_model_name(model_id: str) -> str:
    models = _load_models()["models"]
    info = models.get(model_id)
    return info["name"] if info else model_id
