"""CLI entry point for pi-pods. Uses Click for argument parsing."""

from __future__ import annotations

import asyncio
import subprocess
import sys

import click

from pi.pods.config import get_active_pod, load_config
from pi.pods.ssh import ssh_exec_stream


def _run(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """Manage vLLM deployments on GPU pods."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Pod management
# ---------------------------------------------------------------------------

@main.group(invoke_without_command=True)
@click.pass_context
def pods(ctx):
    """Pod management commands."""
    if ctx.invoked_subcommand is None:
        from pi.pods.commands.pods import list_pods
        list_pods()


@pods.command("setup")
@click.argument("name")
@click.argument("ssh_cmd")
@click.option("--mount", default=None, help="Mount command to run on the pod")
@click.option("--models-path", default=None, help="Path on pod for model storage")
@click.option(
    "--vllm",
    type=click.Choice(["release", "nightly", "gpt-oss"]),
    default="release",
    help="vLLM version to install",
)
def pods_setup(name, ssh_cmd, mount, models_path, vllm):
    """Setup a new pod."""
    from pi.pods.commands.pods import setup_pod
    _run(setup_pod(name, ssh_cmd, mount=mount, models_path=models_path, vllm=vllm))


@pods.command("active")
@click.argument("name")
def pods_active(name):
    """Switch active pod."""
    from pi.pods.commands.pods import switch_active_pod
    switch_active_pod(name)


@pods.command("remove")
@click.argument("name")
def pods_remove(name):
    """Remove a pod from local config."""
    from pi.pods.commands.pods import remove_pod_command
    remove_pod_command(name)


# ---------------------------------------------------------------------------
# SSH / Shell
# ---------------------------------------------------------------------------

@main.command()
@click.argument("name", required=False)
def shell(name):
    """Open an interactive shell on a pod."""
    pod_info = None

    if name:
        config = load_config()
        pod = config.pods.get(name)
        if pod:
            pod_info = (name, pod)
    else:
        pod_info = get_active_pod()

    if not pod_info:
        if name:
            click.echo(f"Pod '{name}' not found", err=True)
        else:
            click.echo("No active pod. Use 'pi-pods pods active <name>' to set one.", err=True)
        sys.exit(1)

    pod_name, pod = pod_info
    click.echo(f"Connecting to pod '{pod_name}'...")

    ssh_args = pod.ssh.split()[1:]  # Remove 'ssh' from command
    proc = subprocess.Popen(
        ["ssh", *ssh_args],
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    sys.exit(proc.wait() or 0)


@main.command()
@click.argument("args", nargs=-1, required=True)
def ssh(args):
    """Run an SSH command on a pod. Usage: pi-pods ssh [<name>] "<command>" """
    if len(args) == 1:
        pod_name_arg = None
        command = args[0]
    elif len(args) == 2:
        pod_name_arg = args[0]
        command = args[1]
    else:
        click.echo('Usage: pi-pods ssh [<name>] "<command>"', err=True)
        sys.exit(1)

    pod_info = None
    if pod_name_arg:
        config = load_config()
        pod = config.pods.get(pod_name_arg)
        if pod:
            pod_info = (pod_name_arg, pod)
    else:
        pod_info = get_active_pod()

    if not pod_info:
        if pod_name_arg:
            click.echo(f"Pod '{pod_name_arg}' not found", err=True)
        else:
            click.echo("No active pod. Use 'pi-pods pods active <name>' to set one.", err=True)
        sys.exit(1)

    pod_name, pod = pod_info
    click.echo(f"Running on pod '{pod_name}': {command}")

    exit_code = _run(ssh_exec_stream(pod.ssh, command))
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

@main.command()
@click.argument("model_id", required=False)
@click.option("--name", required=False, help="Name for the model deployment")
@click.option("--pod", "pod_override", default=None, help="Override active pod")
@click.option("--memory", default=None, help="GPU memory allocation (e.g., 50%%)")
@click.option("--context", default=None, help="Context window size (4k, 8k, 16k, 32k, 64k, 128k)")
@click.option("--gpus", "gpus_count", type=int, default=None, help="Number of GPUs (predefined models only)")
@click.option("--vllm", "vllm_args", multiple=True, help="Custom vLLM arguments")
def start(model_id, name, pod_override, memory, context, gpus_count, vllm_args):
    """Start a model on a pod."""
    from pi.pods.commands.models import show_known_models, start_model

    if not model_id:
        _run(show_known_models())
        return

    if not name:
        click.echo("--name is required", err=True)
        sys.exit(1)

    if vllm_args and (memory or context or gpus_count):
        click.echo("Warning: --memory, --context, and --gpus are ignored when --vllm is specified")
        click.echo("  Using only custom vLLM arguments")
        click.echo()

    _run(start_model(
        model_id,
        name,
        pod_override=pod_override,
        memory=memory,
        context=context,
        gpus_count=gpus_count,
        vllm_args=list(vllm_args) if vllm_args else None,
    ))


@main.command()
@click.argument("name", required=False)
@click.option("--pod", "pod_override", default=None, help="Override active pod")
def stop(name, pod_override):
    """Stop a model (or all models if no name given)."""
    from pi.pods.commands.models import stop_all_models, stop_model

    if not name:
        _run(stop_all_models(pod_override=pod_override))
    else:
        _run(stop_model(name, pod_override=pod_override))


@main.command("list")
@click.option("--pod", "pod_override", default=None, help="Override active pod")
def list_cmd(pod_override):
    """List running models."""
    from pi.pods.commands.models import list_models
    _run(list_models(pod_override=pod_override))


@main.command()
@click.argument("name")
@click.option("--pod", "pod_override", default=None, help="Override active pod")
def logs(name, pod_override):
    """Stream model logs."""
    from pi.pods.commands.models import view_logs
    _run(view_logs(name, pod_override=pod_override))


@main.command()
@click.argument("name")
@click.argument("user_args", nargs=-1)
@click.option("--pod", "pod_override", default=None, help="Override active pod")
def agent(name, user_args, pod_override):
    """Chat with a deployed model using the agent."""
    from pi.pods.commands.prompt import prompt_model

    api_key = None
    import os
    api_key = os.environ.get("PI_API_KEY")

    _run(prompt_model(name, list(user_args), pod=pod_override, api_key=api_key))


if __name__ == "__main__":
    main()
