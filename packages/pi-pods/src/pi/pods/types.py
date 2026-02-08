"""Core type definitions for pi-pods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GPU:
    id: int
    name: str
    memory: str


@dataclass
class Model:
    model: str
    port: int
    gpu: list[int]
    pid: int


@dataclass
class Pod:
    ssh: str
    gpus: list[GPU] = field(default_factory=list)
    models: dict[str, Model] = field(default_factory=dict)
    models_path: str | None = None
    vllm_version: Literal["release", "nightly", "gpt-oss"] | None = None


@dataclass
class Config:
    pods: dict[str, Pod] = field(default_factory=dict)
    active: str | None = None


def pod_from_dict(data: dict) -> Pod:
    """Deserialize a Pod from a JSON-compatible dict."""
    gpus = [GPU(**g) for g in data.get("gpus", [])]
    models = {
        name: Model(**m) for name, m in data.get("models", {}).items()
    }
    return Pod(
        ssh=data["ssh"],
        gpus=gpus,
        models=models,
        models_path=data.get("modelsPath"),
        vllm_version=data.get("vllmVersion"),
    )


def pod_to_dict(pod: Pod) -> dict:
    """Serialize a Pod to a JSON-compatible dict."""
    return {
        "ssh": pod.ssh,
        "gpus": [{"id": g.id, "name": g.name, "memory": g.memory} for g in pod.gpus],
        "models": {
            name: {"model": m.model, "port": m.port, "gpu": m.gpu, "pid": m.pid}
            for name, m in pod.models.items()
        },
        "modelsPath": pod.models_path,
        "vllmVersion": pod.vllm_version,
    }


def config_from_dict(data: dict) -> Config:
    """Deserialize a Config from a JSON-compatible dict."""
    pods = {name: pod_from_dict(p) for name, p in data.get("pods", {}).items()}
    return Config(pods=pods, active=data.get("active"))


def config_to_dict(config: Config) -> dict:
    """Serialize a Config to a JSON-compatible dict."""
    return {
        "pods": {name: pod_to_dict(p) for name, p in config.pods.items()},
        "active": config.active,
    }
