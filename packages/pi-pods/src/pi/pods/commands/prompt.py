"""Agent prompt command for chatting with deployed models."""

from __future__ import annotations

import os
import sys

from pi.pods.config import get_active_pod, load_config


async def prompt_model(
    model_name: str,
    user_args: list[str],
    *,
    pod: str | None = None,
    api_key: str | None = None,
) -> None:
    if pod:
        config = load_config()
        pod_obj = config.pods.get(pod)
        if not pod_obj:
            print(f"Pod '{pod}' not found", file=sys.stderr)
            sys.exit(1)
        pod_name = pod
    else:
        active = get_active_pod()
        if not active:
            print("No active pod. Use 'pi-pods pods active <name>' to set one.", file=sys.stderr)
            sys.exit(1)
        pod_name, pod_obj = active

    model_config = pod_obj.models.get(model_name)
    if not model_config:
        print(f"Model '{model_name}' not found on pod '{pod_name}'", file=sys.stderr)
        sys.exit(1)

    # Extract host from SSH string
    host = "localhost"
    for part in pod_obj.ssh.split():
        if "@" in part:
            host = part.split("@")[1]
            break

    _system_prompt = (
        "You help the user understand and navigate the codebase in the current working directory.\n\n"
        "You can read files, list directories, and execute shell commands via the respective tools.\n\n"
        "Do not output file contents you read via the read_file tool directly, unless asked to.\n\n"
        "Do not output markdown tables as part of your responses.\n\n"
        "Keep your responses concise and relevant to the user's request.\n\n"
        "File paths you output must include line numbers where possible, "
        'e.g. "src/index.ts:10-20" for lines 10 to 20 in src/index.ts.\n\n'
        f"Current working directory: {os.getcwd()}"
    )

    _base_url = f"http://{host}:{model_config.port}/v1"
    _api = "responses" if "gpt-oss" in model_config.model.lower() else "completions"
    _key = api_key or os.environ.get("PI_API_KEY", "dummy")

    # Build arguments for agent main function
    _args = [
        "--base-url", _base_url,
        "--model", model_config.model,
        "--api-key", _key,
        "--api", _api,
        "--system-prompt", _system_prompt,
        *user_args,
    ]

    # TODO: Call agent main function directly once pi-agent CLI integration is available
    raise NotImplementedError("Agent prompt not yet implemented")
