"""pi-pods: GPU pod manager for vLLM deployments."""

from pi.pods.config import (
    add_pod,
    get_active_pod,
    load_config,
    remove_pod,
    save_config,
    set_active_pod,
)
from pi.pods.model_configs import (
    get_known_models,
    get_model_config,
    get_model_name,
    is_known_model,
)
from pi.pods.ssh import SSHResult, scp_file, ssh_exec, ssh_exec_stream
from pi.pods.types import GPU, Config, Model, Pod

__all__ = [
    # Types
    "GPU",
    "Model",
    "Pod",
    "Config",
    # Config
    "load_config",
    "save_config",
    "get_active_pod",
    "add_pod",
    "remove_pod",
    "set_active_pod",
    # SSH
    "SSHResult",
    "ssh_exec",
    "ssh_exec_stream",
    "scp_file",
    # Model configs
    "get_model_config",
    "is_known_model",
    "get_known_models",
    "get_model_name",
]
