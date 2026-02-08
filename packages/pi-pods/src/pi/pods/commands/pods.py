"""Pod management commands: setup, list, switch, remove."""

from __future__ import annotations

import os
import sys

from pi.pods.config import add_pod, load_config, remove_pod, set_active_pod
from pi.pods.ssh import scp_file, ssh_exec, ssh_exec_stream
from pi.pods.types import GPU, Pod


def list_pods() -> None:
    config = load_config()
    pod_names = list(config.pods.keys())

    if not pod_names:
        print("No pods configured. Use 'pi-pods pods setup' to add a pod.")
        return

    print("Configured pods:")
    for name in pod_names:
        pod = config.pods[name]
        is_active = config.active == name
        marker = "*" if is_active else " "
        gpu_count = len(pod.gpus)
        gpu_info = f"{gpu_count}x {pod.gpus[0].name}" if gpu_count > 0 else "no GPUs detected"
        vllm_info = f" (vLLM: {pod.vllm_version})" if pod.vllm_version else ""
        print(f"{marker} {name} - {gpu_info}{vllm_info} - {pod.ssh}")
        if pod.models_path:
            print(f"    Models: {pod.models_path}")
        if pod.vllm_version == "gpt-oss":
            print("    WARNING: GPT-OSS build - only for GPT-OSS models")


async def setup_pod(
    name: str,
    ssh_cmd: str,
    *,
    mount: str | None = None,
    models_path: str | None = None,
    vllm: str = "release",
) -> None:
    # Validate environment variables
    hf_token = os.environ.get("HF_TOKEN")
    vllm_api_key = os.environ.get("PI_API_KEY")

    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is required", file=sys.stderr)
        print("Get a token from: https://huggingface.co/settings/tokens", file=sys.stderr)
        print("Then run: export HF_TOKEN=your_token_here", file=sys.stderr)
        sys.exit(1)

    if not vllm_api_key:
        print("ERROR: PI_API_KEY environment variable is required", file=sys.stderr)
        print("Set an API key: export PI_API_KEY=your_api_key_here", file=sys.stderr)
        sys.exit(1)

    # Determine models path
    if not models_path and mount:
        parts = mount.split(" ")
        models_path = parts[-1]

    if not models_path:
        print("ERROR: --models-path is required (or must be extractable from --mount)", file=sys.stderr)
        sys.exit(1)

    print(f"Setting up pod '{name}'...")
    print(f"SSH: {ssh_cmd}")
    print(f"Models path: {models_path}")
    vllm_label = f"{vllm} (GPT-OSS special build)" if vllm == "gpt-oss" else vllm
    print(f"vLLM version: {vllm_label}")
    if mount:
        print(f"Mount command: {mount}")
    print()

    # Test SSH connection
    print("Testing SSH connection...")
    test_result = await ssh_exec(ssh_cmd, "echo 'SSH OK'")
    if test_result.exit_code != 0:
        print("Failed to connect via SSH", file=sys.stderr)
        print(test_result.stderr, file=sys.stderr)
        sys.exit(1)
    print("SSH connection successful")

    # Copy setup script
    print("Copying setup script...")
    from pathlib import Path
    script_path = str(Path(__file__).parent.parent / "scripts" / "pod_setup.sh")
    success = await scp_file(ssh_cmd, script_path, "/tmp/pod_setup.sh")
    if not success:
        print("Failed to copy setup script", file=sys.stderr)
        sys.exit(1)
    print("Setup script copied")

    # Build setup command
    setup_cmd = f"bash /tmp/pod_setup.sh --models-path '{models_path}' --hf-token '{hf_token}' --vllm-api-key '{vllm_api_key}'"
    if mount:
        setup_cmd += f" --mount '{mount}'"
    setup_cmd += f" --vllm '{vllm}'"

    # Run setup script
    print()
    print("Running setup (this will take 2-5 minutes)...")
    print()

    exit_code = await ssh_exec_stream(ssh_cmd, setup_cmd, force_tty=True)
    if exit_code != 0:
        print("\nSetup failed. Check the output above for errors.", file=sys.stderr)
        sys.exit(1)

    # Detect GPU configuration
    print()
    print("Detecting GPU configuration...")
    gpu_result = await ssh_exec(
        ssh_cmd,
        "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader",
    )

    gpus: list[GPU] = []
    if gpu_result.exit_code == 0 and gpu_result.stdout:
        for line in gpu_result.stdout.strip().split("\n"):
            parts_list = [s.strip() for s in line.split(",")]
            if len(parts_list) >= 1 and parts_list[0]:
                gpus.append(GPU(
                    id=int(parts_list[0]),
                    name=parts_list[1] if len(parts_list) > 1 else "Unknown",
                    memory=parts_list[2] if len(parts_list) > 2 else "Unknown",
                ))

    print(f"Detected {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"  GPU {gpu.id}: {gpu.name} ({gpu.memory})")

    pod = Pod(
        ssh=ssh_cmd,
        gpus=gpus,
        models={},
        models_path=models_path,
        vllm_version=vllm,  # type: ignore[arg-type]
    )

    add_pod(name, pod)
    print()
    print(f"Pod '{name}' setup complete and set as active pod")
    print()
    print("You can now deploy models with:")
    print(f"  pi-pods start <model> --name <name>")


def switch_active_pod(name: str) -> None:
    config = load_config()
    if name not in config.pods:
        print(f"Pod '{name}' not found", file=sys.stderr)
        print("\nAvailable pods:")
        for pod_name in config.pods:
            print(f"  {pod_name}")
        sys.exit(1)

    set_active_pod(name)
    print(f"Switched active pod to '{name}'")


def remove_pod_command(name: str) -> None:
    config = load_config()
    if name not in config.pods:
        print(f"Pod '{name}' not found", file=sys.stderr)
        sys.exit(1)

    remove_pod(name)
    print(f"Removed pod '{name}' from configuration")
    print("Note: This only removes the local configuration. The remote pod is not affected.")
