# pi-pods

GPU pod manager for self-hosted vLLM deployments. Handles pod setup, model deployment, GPU allocation, and log streaming over SSH.

## Running

```bash
pi-pods --help
```

## Architecture

```
CLI (click)
  |-- pods: setup, list, switch, remove
  |-- models: start, stop, list, logs
  |-- shell/ssh: interactive access
  |
Config (~/.pi/pods.json)
  |-- Pod definitions (SSH, GPUs, models)
  |-- Active pod tracking
  |
SSH (subprocess)
  |-- Command execution (ssh)
  |-- File transfer (scp)
  |-- Log streaming (tail -f)
  |
Model Configs (models.json)
  |-- Hardware-aware vLLM argument selection
  |-- GPU type/count matching
```

## Commands

### Pod Management

```bash
# Setup a new pod with GPU detection
pi-pods pods setup my-pod "ssh root@1.2.3.4" --mount "mount -t nfs ... /mnt/data"

# List all configured pods
pi-pods pods

# Switch active pod
pi-pods pods active my-pod

# Remove a pod from local config
pi-pods pods remove my-pod

# Open interactive shell
pi-pods shell

# Run remote command
pi-pods ssh "nvidia-smi"
```

### Model Deployment

```bash
# Start a known model (auto-selects GPU config)
pi-pods start Qwen/Qwen2.5-Coder-32B-Instruct --name qwen32b

# Start with specific GPU count
pi-pods start Qwen/Qwen3-Coder-30B-A3B-Instruct --name qwen3 --gpus 2

# Start with memory/context overrides
pi-pods start Qwen/Qwen2.5-Coder-32B-Instruct --name qwen32b --memory 50% --context 16k

# Start with custom vLLM arguments
pi-pods start my/model --name custom --vllm --tensor-parallel-size 4

# Stop a model
pi-pods stop qwen32b

# Stop all models
pi-pods stop

# List running models with health check
pi-pods list

# Stream model logs
pi-pods logs qwen32b

# Show known models and compatibility
pi-pods start
```

## Configuration

Config is stored at `~/.pi/pods.json` (or `$PI_CONFIG_DIR/pods.json`).

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token for model downloads |
| `PI_API_KEY` | API key for vLLM endpoints |
| `PI_CONFIG_DIR` | Config directory (default: `~/.pi`) |

## Model Configs

The `models.json` file contains hardware-aware configurations for known models. Each model has one or more configs specifying GPU count, compatible GPU types, and vLLM arguments.

When you run `pi-pods start <model>`, the tool finds the best matching config for your pod's hardware and applies the correct vLLM arguments automatically.

## File Structure

```
src/pi/pods/
    __init__.py           Public exports
    types.py              Core dataclasses (GPU, Model, Pod, Config)
    config.py             Config load/save (~/.pi/pods.json)
    ssh.py                SSH execution via subprocess
    model_configs.py      Model config matching from models.json
    cli.py                Click CLI entry point
    models.json           Model hardware configurations
    commands/
        models.py         Model start/stop/list/logs
        pods.py           Pod setup/list/switch/remove
        prompt.py         Agent prompt (stub)
    scripts/
        model_run.sh      Model runner script (uploaded to pod)
        pod_setup.sh      Pod bootstrap script (uploaded to pod)
```
