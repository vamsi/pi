"""Model management commands: start, stop, list, logs, known models."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from pi.pods.config import get_active_pod, load_config, save_config
from pi.pods.model_configs import get_model_config, get_model_name, is_known_model
from pi.pods.ssh import ssh_exec
from pi.pods.types import Pod


def _get_pod(pod_override: str | None = None) -> tuple[str, Pod]:
    """Get the pod to use (active or override)."""
    if pod_override:
        config = load_config()
        pod = config.pods.get(pod_override)
        if not pod:
            print(f"Pod '{pod_override}' not found", file=sys.stderr)
            sys.exit(1)
        return (pod_override, pod)

    active = get_active_pod()
    if not active:
        print("No active pod. Use 'pi-pods pods active <name>' to set one.", file=sys.stderr)
        sys.exit(1)
    return active


def _get_next_port(pod: Pod) -> int:
    """Find next available port starting from 8001."""
    used_ports = {m.port for m in pod.models.values()}
    port = 8001
    while port in used_ports:
        port += 1
    return port


def _select_gpus(pod: Pod, count: int = 1) -> list[int]:
    """Select GPUs for model deployment (round-robin, least-used first)."""
    if count == len(pod.gpus):
        return [g.id for g in pod.gpus]

    # Count GPU usage across all models
    gpu_usage: dict[int, int] = {g.id: 0 for g in pod.gpus}
    for model in pod.models.values():
        for gpu_id in model.gpu:
            gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1

    # Sort by usage (least used first)
    sorted_gpus = sorted(gpu_usage.items(), key=lambda x: x[1])
    return [gpu_id for gpu_id, _ in sorted_gpus[:count]]


def _get_host(pod: Pod) -> str:
    """Extract host from SSH string."""
    for part in pod.ssh.split():
        if "@" in part:
            return part.split("@")[1]
    return "localhost"


async def start_model(
    model_id: str,
    name: str,
    *,
    pod_override: str | None = None,
    vllm_args: list[str] | None = None,
    memory: str | None = None,
    context: str | None = None,
    gpus_count: int | None = None,
) -> None:
    pod_name, pod = _get_pod(pod_override)

    if not pod.models_path:
        print("Pod does not have a models path configured", file=sys.stderr)
        sys.exit(1)
    if name in pod.models:
        print(f"Model '{name}' already exists on pod '{pod_name}'", file=sys.stderr)
        sys.exit(1)

    port = _get_next_port(pod)

    gpus: list[int] = []
    args: list[str] = []
    model_config = None

    if vllm_args:
        args = list(vllm_args)
        print("Using custom vLLM args, GPU allocation managed by vLLM")
    elif is_known_model(model_id):
        if gpus_count is not None:
            if gpus_count > len(pod.gpus):
                print(f"Error: Requested {gpus_count} GPUs but pod only has {len(pod.gpus)}", file=sys.stderr)
                sys.exit(1)

            model_config = get_model_config(model_id, pod.gpus, gpus_count)
            if model_config:
                gpus = _select_gpus(pod, gpus_count)
                args = list(model_config.get("args", []))
            else:
                print(
                    f"Model '{get_model_name(model_id)}' does not have a configuration for {gpus_count} GPU(s)",
                    file=sys.stderr,
                )
                print("Available configurations:", file=sys.stderr)
                for gc in range(1, len(pod.gpus) + 1):
                    cfg = get_model_config(model_id, pod.gpus, gc)
                    if cfg:
                        print(f"  - {gc} GPU(s)", file=sys.stderr)
                sys.exit(1)
        else:
            for gc in range(len(pod.gpus), 0, -1):
                model_config = get_model_config(model_id, pod.gpus, gc)
                if model_config:
                    gpus = _select_gpus(pod, gc)
                    args = list(model_config.get("args", []))
                    break
            if not model_config:
                print(f"Model '{get_model_name(model_id)}' not compatible with this pod's GPUs", file=sys.stderr)
                sys.exit(1)
    else:
        if gpus_count is not None:
            print("Error: --gpus can only be used with predefined models", file=sys.stderr)
            print("For custom models, use --vllm with tensor-parallel-size or similar arguments", file=sys.stderr)
            sys.exit(1)
        gpus = _select_gpus(pod, 1)
        print("Unknown model, defaulting to single GPU")

    # Apply memory/context overrides
    if not vllm_args:
        if memory:
            fraction = float(memory.replace("%", "")) / 100
            args = [a for a in args if "gpu-memory-utilization" not in a]
            args.extend(["--gpu-memory-utilization", str(fraction)])
        if context:
            context_sizes = {
                "4k": 4096, "8k": 8192, "16k": 16384,
                "32k": 32768, "64k": 65536, "128k": 131072,
            }
            max_tokens = context_sizes.get(context.lower(), 0)
            if not max_tokens:
                try:
                    max_tokens = int(context)
                except ValueError:
                    pass
            if max_tokens:
                args = [a for a in args if "max-model-len" not in a]
                args.extend(["--max-model-len", str(max_tokens)])

    print(f"Starting model '{name}' on pod '{pod_name}'...")
    print(f"Model: {model_id}")
    print(f"Port: {port}")
    print(f"GPU(s): {', '.join(str(g) for g in gpus) if gpus else 'Managed by vLLM'}")
    if model_config and model_config.get("notes"):
        print(f"Note: {model_config['notes']}")
    print()

    # Read and customize model_run.sh script
    script_path = Path(__file__).parent.parent / "scripts" / "model_run.sh"
    script_content = script_path.read_text()

    script_content = (
        script_content
        .replace("{{MODEL_ID}}", model_id)
        .replace("{{NAME}}", name)
        .replace("{{PORT}}", str(port))
        .replace("{{VLLM_ARGS}}", " ".join(args))
    )

    # Upload customized script
    await ssh_exec(
        pod.ssh,
        f"cat > /tmp/model_run_{name}.sh << 'EOF'\n{script_content}\nEOF\nchmod +x /tmp/model_run_{name}.sh",
    )

    # Prepare environment
    env_lines = [
        f"export HF_TOKEN='{os.environ.get('HF_TOKEN', '')}'",
        f"export PI_API_KEY='{os.environ.get('PI_API_KEY', '')}'",
        "export HF_HUB_ENABLE_HF_TRANSFER=1",
        "export VLLM_NO_USAGE_STATS=1",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "export FORCE_COLOR=1",
        "export TERM=xterm-256color",
    ]
    if len(gpus) == 1:
        env_lines.append(f"export CUDA_VISIBLE_DEVICES={gpus[0]}")
    if model_config and model_config.get("env"):
        for k, v in model_config["env"].items():
            env_lines.append(f"export {k}='{v}'")

    env_str = "\n".join(env_lines)

    start_cmd = f"""
        {env_str}
        mkdir -p ~/.vllm_logs
        cat > /tmp/model_wrapper_{name}.sh << 'WRAPPER'
#!/bin/bash
script -q -f -c "/tmp/model_run_{name}.sh" ~/.vllm_logs/{name}.log
exit_code=$?
echo "Script exited with code $exit_code" >> ~/.vllm_logs/{name}.log
exit $exit_code
WRAPPER
        chmod +x /tmp/model_wrapper_{name}.sh
        setsid /tmp/model_wrapper_{name}.sh </dev/null >/dev/null 2>&1 &
        echo $!
        exit 0
    """

    pid_result = await ssh_exec(pod.ssh, start_cmd)
    pid_str = pid_result.stdout.strip()
    try:
        pid = int(pid_str)
    except ValueError:
        print("Failed to start model runner", file=sys.stderr)
        sys.exit(1)

    # Save to config
    config = load_config()
    from pi.pods.types import Model as PodModel
    config.pods[pod_name].models[name] = PodModel(model=model_id, port=port, gpu=gpus, pid=pid)
    save_config(config)

    print(f"Model runner started with PID: {pid}")
    print("Streaming logs... (waiting for startup)\n")

    await asyncio.sleep(0.5)

    # Stream logs, watching for startup complete
    host = _get_host(pod)
    ssh_parts = pod.ssh.split()
    ssh_binary = ssh_parts[0]
    ssh_args = ssh_parts[1:]
    tail_cmd = f"tail -f ~/.vllm_logs/{name}.log"

    proc = subprocess.Popen(
        [ssh_binary, *ssh_args, tail_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "FORCE_COLOR": "1"},
    )

    interrupted = False
    startup_complete = False
    startup_failed = False
    failure_reason = ""

    original_sigint = signal.getsignal(signal.SIGINT)

    def sigint_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        proc.kill()

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        for stream in [proc.stdout, proc.stderr]:
            if not stream:
                continue
            while True:
                line_bytes = stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace").rstrip("\n")
                if line:
                    print(line)

                    if "Application startup complete" in line:
                        startup_complete = True
                        proc.kill()
                        break

                    if ("Model runner exiting with code" in line and "code 0" not in line):
                        startup_failed = True
                        failure_reason = "Model runner failed to start"
                        proc.kill()
                        break
                    if ("Script exited with code" in line and "code 0" not in line):
                        startup_failed = True
                        failure_reason = "Script failed to execute"
                        proc.kill()
                        break
                    if "torch.OutOfMemoryError" in line or "CUDA out of memory" in line:
                        startup_failed = True
                        failure_reason = "Out of GPU memory (OOM)"
                    if "RuntimeError: Engine core initialization failed" in line:
                        startup_failed = True
                        failure_reason = "vLLM engine initialization failed"
                        proc.kill()
                        break

                if startup_complete or startup_failed or interrupted:
                    break
            if startup_complete or startup_failed or interrupted:
                break

        proc.wait()
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    if startup_failed:
        print(f"\nModel failed to start: {failure_reason}")

        config = load_config()
        config.pods[pod_name].models.pop(name, None)
        save_config(config)

        print("\nModel has been removed from configuration.")

        if "OOM" in failure_reason or "memory" in failure_reason:
            print("\nSuggestions:")
            print("  - Try reducing GPU memory utilization: --memory 50%")
            print("  - Use a smaller context window: --context 4k")
            print("  - Use a quantized version of the model (e.g., FP8)")
            print("  - Use more GPUs with tensor parallelism")
            print("  - Try a smaller model variant")

        print(f'\nCheck full logs: pi-pods ssh "tail -100 ~/.vllm_logs/{name}.log"')
        sys.exit(1)

    elif startup_complete:
        print(f"\nModel started successfully!")
        print(f"\nConnection Details:")
        print("-" * 50)
        print(f"Base URL:    http://{host}:{port}/v1")
        print(f"Model:       {model_id}")
        print(f"API Key:     {os.environ.get('PI_API_KEY', '(not set)')}")
        print("-" * 50)

        print(f"\nExport for shell:")
        print(f'export OPENAI_BASE_URL="http://{host}:{port}/v1"')
        print(f'export OPENAI_API_KEY="{os.environ.get("PI_API_KEY", "your-api-key")}"')
        print(f'export OPENAI_MODEL="{model_id}"')

        print(f"\nExample usage:")
        print(f"""
  # Python
  from openai import OpenAI
  client = OpenAI()  # Uses env vars
  response = client.chat.completions.create(
      model="{model_id}",
      messages=[{{"role": "user", "content": "Hello!"}}]
  )

  # CLI
  curl $OPENAI_BASE_URL/chat/completions \\
    -H "Authorization: Bearer $OPENAI_API_KEY" \\
    -H "Content-Type: application/json" \\
    -d '{{"model":"{model_id}","messages":[{{"role":"user","content":"Hi"}}]}}'""")
        print()
        print(f"Chat with model:  pi-pods agent {name} \"Your message\"")
        print(f"Monitor logs:     pi-pods logs {name}")
        print(f"Stop model:       pi-pods stop {name}")

    elif interrupted:
        print("\n\nStopped monitoring. Model deployment continues in background.")
        print(f"Chat with model: pi-pods agent {name} \"Your message\"")
        print(f"Check status: pi-pods logs {name}")
        print(f"Stop model: pi-pods stop {name}")

    else:
        print("\n\nLog stream ended. Model may still be running.")
        print(f"Chat with model: pi-pods agent {name} \"Your message\"")
        print(f"Check status: pi-pods logs {name}")
        print(f"Stop model: pi-pods stop {name}")


async def stop_model(name: str, *, pod_override: str | None = None) -> None:
    pod_name, pod = _get_pod(pod_override)

    model = pod.models.get(name)
    if not model:
        print(f"Model '{name}' not found on pod '{pod_name}'", file=sys.stderr)
        sys.exit(1)

    print(f"Stopping model '{name}' on pod '{pod_name}'...")

    kill_cmd = f"""
        pkill -TERM -P {model.pid} 2>/dev/null || true
        kill {model.pid} 2>/dev/null || true
    """
    await ssh_exec(pod.ssh, kill_cmd)

    config = load_config()
    config.pods[pod_name].models.pop(name, None)
    save_config(config)

    print(f"Model '{name}' stopped")


async def stop_all_models(*, pod_override: str | None = None) -> None:
    pod_name, pod = _get_pod(pod_override)

    model_names = list(pod.models.keys())
    if not model_names:
        print(f"No models running on pod '{pod_name}'")
        return

    print(f"Stopping {len(model_names)} model(s) on pod '{pod_name}'...")

    pids = [str(m.pid) for m in pod.models.values()]
    kill_cmd = f"""
        for PID in {' '.join(pids)}; do
            pkill -TERM -P $PID 2>/dev/null || true
            kill $PID 2>/dev/null || true
        done
    """
    await ssh_exec(pod.ssh, kill_cmd)

    config = load_config()
    config.pods[pod_name].models = {}
    save_config(config)

    print(f"Stopped all models: {', '.join(model_names)}")


async def list_models(*, pod_override: str | None = None) -> None:
    pod_name, pod = _get_pod(pod_override)

    model_names = list(pod.models.keys())
    if not model_names:
        print(f"No models running on pod '{pod_name}'")
        return

    host = _get_host(pod)

    print(f"Models on pod '{pod_name}':")
    for mname in model_names:
        model = pod.models[mname]
        if len(model.gpu) > 1:
            gpu_str = f"GPUs {','.join(str(g) for g in model.gpu)}"
        elif len(model.gpu) == 1:
            gpu_str = f"GPU {model.gpu[0]}"
        else:
            gpu_str = "GPU unknown"
        print(f"  {mname} - Port {model.port} - {gpu_str} - PID {model.pid}")
        print(f"    Model: {model.model}")
        print(f"    URL: http://{host}:{model.port}/v1")

    # Verify processes
    print()
    print("Verifying processes...")
    any_dead = False
    for mname in model_names:
        model = pod.models[mname]
        check_cmd = f"""
            if ps -p {model.pid} > /dev/null 2>&1; then
                if curl -s -f http://localhost:{model.port}/health > /dev/null 2>&1; then
                    echo "running"
                else
                    if tail -n 20 ~/.vllm_logs/{mname}.log 2>/dev/null | grep -q "ERROR\\|Failed\\|Cuda error\\|died"; then
                        echo "crashed"
                    else
                        echo "starting"
                    fi
                fi
            else
                echo "dead"
            fi
        """
        result = await ssh_exec(pod.ssh, check_cmd)
        status = result.stdout.strip()
        if status == "dead":
            print(f"  {mname}: Process {model.pid} is not running")
            any_dead = True
        elif status == "crashed":
            print(f"  {mname}: vLLM crashed (check logs with 'pi-pods logs {mname}')")
            any_dead = True
        elif status == "starting":
            print(f"  {mname}: Still starting up...")

    if any_dead:
        print()
        print("Some models are not running. Clean up with:")
        print("  pi-pods stop <name>")
    else:
        print("All processes verified")


async def view_logs(name: str, *, pod_override: str | None = None) -> None:
    pod_name, pod = _get_pod(pod_override)

    model = pod.models.get(name)
    if not model:
        print(f"Model '{name}' not found on pod '{pod_name}'", file=sys.stderr)
        sys.exit(1)

    print(f"Streaming logs for '{name}' on pod '{pod_name}'...")
    print("Press Ctrl+C to stop")
    print()

    ssh_parts = pod.ssh.split()
    ssh_binary = ssh_parts[0]
    ssh_args = ssh_parts[1:]
    tail_cmd = f"tail -f ~/.vllm_logs/{name}.log"

    proc = subprocess.Popen(
        [ssh_binary, *ssh_args, tail_cmd],
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={**os.environ, "FORCE_COLOR": "1"},
    )

    proc.wait()


async def show_known_models() -> None:
    models_json_path = Path(__file__).parent.parent / "models.json"
    models_data = json.loads(models_json_path.read_text())
    models = models_data["models"]

    active = get_active_pod()
    pod_gpu_count = 0
    pod_gpu_type = ""

    if active:
        _, active_pod = active
        pod_gpu_count = len(active_pod.gpus)
        if active_pod.gpus:
            pod_gpu_type = active_pod.gpus[0].name.replace("NVIDIA", "").strip().split(" ")[0]
        print(f"Known Models for {active[0]} ({pod_gpu_count}x {pod_gpu_type or 'GPU'}):\n")
    else:
        print("Known Models:\n")
        print("No active pod. Use 'pi-pods pods active <name>' to filter compatible models.\n")

    print("Usage: pi-pods start <model> --name <name> [options]\n")

    compatible: dict[str, list[dict]] = {}
    incompatible: dict[str, list[dict]] = {}

    for model_id, info in models.items():
        family = info["name"].split("-")[0] or "Other"

        is_compatible = False
        compatible_config = ""
        min_gpu = "Unknown"
        min_notes = None

        configs = info.get("configs", [])
        if configs:
            sorted_configs = sorted(configs, key=lambda c: c.get("gpuCount", 1))

            min_config = sorted_configs[0]
            min_gpu_count = min_config.get("gpuCount", 1)
            gpu_types = "/".join(min_config.get("gpuTypes", ["H100/H200"]))
            min_gpu = f"{min_gpu_count}x {gpu_types}"
            min_notes = min_config.get("notes") or info.get("notes")

            if active and pod_gpu_count > 0:
                for cfg in sorted_configs:
                    cfg_gpu_count = cfg.get("gpuCount", 1)
                    cfg_gpu_types = cfg.get("gpuTypes", [])

                    if cfg_gpu_count <= pod_gpu_count:
                        if not cfg_gpu_types or any(
                            pod_gpu_type in t or t in pod_gpu_type
                            for t in cfg_gpu_types
                        ):
                            is_compatible = True
                            compatible_config = f"{cfg_gpu_count}x {pod_gpu_type}"
                            min_notes = cfg.get("notes") or info.get("notes")
                            break

        entry = {"id": model_id, "name": info["name"], "notes": min_notes}

        if active and is_compatible:
            compatible.setdefault(family, []).append({**entry, "config": compatible_config})
        else:
            incompatible.setdefault(family, []).append({**entry, "minGpu": min_gpu})

    if active and compatible:
        print("Compatible Models:\n")
        for family in sorted(compatible):
            print(f"{family} Models:")
            for model in sorted(compatible[family], key=lambda m: m["name"]):
                print(f"  {model['id']}")
                print(f"    Name: {model['name']}")
                print(f"    Config: {model['config']}")
                if model.get("notes"):
                    print(f"    Note: {model['notes']}")
                print()

    if incompatible:
        if active and compatible:
            print("Incompatible Models (need more/different GPUs):\n")

        for family in sorted(incompatible):
            print(f"{family} Models:")
            for model in sorted(incompatible[family], key=lambda m: m["name"]):
                print(f"  {model['id']}")
                print(f"    Name: {model['name']}")
                print(f"    Min Hardware: {model['minGpu']}")
                if model.get("notes") and not active:
                    print(f"    Note: {model['notes']}")
                print()

    print("For unknown models, defaults to single GPU deployment.")
    print("Use --vllm to pass custom arguments to vLLM.")
