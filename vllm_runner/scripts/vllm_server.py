from speedy_utils import fprint, speedy_timer

speedy_timer.start()
import argparse
import os
import signal
import subprocess
from vllm_runner.app import get_required_tp, scanfree_port
from vllm_runner.scan_gpus import free_gpu, scan_available_gpus
from vllm_runner.scan_vllm_process import scan_vllm_process

speedy_timer.update_task("imports")


def get_parser():
    parser = argparse.ArgumentParser(description="Manage vLLM servers")
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start a vLLM server")
    start_parser.add_argument(
        "--model_size", "-z", type=str, help="Model size (e.g., 72B)"
    )
    start_parser.add_argument(
        "--gpus", type=str, required=True, help="GPUs to use (e.g., 0123)"
    )
    start_parser.add_argument(
        "--port", type=int, default=2800, help="Starting port number"
    )
    start_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    # Add LoRA arguments
    start_parser.add_argument(
        "--lora-modules", nargs="+", help="LoRA modules in format: name=path"
    )
    # Add GPU utilization
    start_parser.add_argument(
        "--gpu-util", type=float, default=0.95, help="GPU utilization per server"
    )
    # add max tokens
    start_parser.add_argument(
        "--max-tokens", type=int, default=8000, help="Max tokens per request"
    )
    # Add extra arguments
    start_parser.add_argument(
        "--extra", type=str, help="Extra arguments to pass to the server"
    )

    stop_parser = subparsers.add_parser(
        "stop_at", help="Stop vLLM servers at specified GPUs"
    )
    stop_parser.add_argument(
        "--gpus", type=str, required=True, help="GPUs to free (e.g., 0123)"
    )

    # Add new ls subparser
    subparsers.add_parser("ls", help="List all running vLLM servers")

    # Fix: Add grep-based stop command
    stop_grep_parser = subparsers.add_parser(
        "stop", help="Stop vLLM servers matching pattern"
    )
    stop_grep_parser.add_argument(
        "--grep", type=str, help="Pattern to match against model name"
    )

    args = parser.parse_args()
    # if len(str(args.model_size)) < 10:
    #     # must be number
    #     assert (
    #         args.model_size.isdigit()
    #     ), f"Model size must be a number, got {args.model_size}"
    return parser, args


parser, args = get_parser()


from vllm_runner.app import get_required_tp, scanfree_port
from vllm_runner.scan_gpus import free_gpu, scan_available_gpus
from vllm_runner.scan_vllm_process import scan_vllm_process


def resolve_model_name(size_or_name):
    # TODO make it work for other models
    if str(size_or_name).isdigit():
        return f"Qwen/Qwen2.5-{size_or_name}B-Instruct-AWQ", f"QW{size_or_name}B"
    else:
        return size_or_name, size_or_name.split("/")[-1]


def start_server(
    size_or_name,
    gpus,
    port_start,
    lora_modules=None,
    dry_run=False,
    gpu_util=0.9,
    max_model_len=8000,
    extra_args=None,
):
    model_name, model_serve_name = resolve_model_name(size_or_name)
    try:
        # Ensure GPUs are available
        available_gpus = scan_available_gpus(mem_threshold=gpu_util)
        selected_gpus = [gpu for gpu in gpus if gpu in available_gpus]
        if not selected_gpus:
            print(f"No available GPUs among {gpus}")
            return
        gpu_ids = ",".join(selected_gpus)
        # Find a free port starting from port_start
        available_ports = scanfree_port(start=int(port_start))
        if not available_ports:
            print("No available ports found!")
            return
        port = available_ports[0]
        # Prepare the command to start the server
        tp = len(selected_gpus)

        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_ids} "
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model_name} "
            f"--served-model-name {model_serve_name} "
            f"--tensor-parallel-size {tp} "
            f"--port {port} "
            f"--host 0.0.0.0 "
            f"--gpu-memory-utilization {gpu_util} "
            f"--max-model-len {max_model_len} "
            f"--enable-prefix-caching "
            f"--disable-log-requests"
        )
        if extra_args:
            command += f" {extra_args}"

        # Add LoRA support if modules are specified
        if lora_modules:
            command += " --enable-lora"
            for name, path in lora_modules.items():
                command += f" --lora-modules {name}={path}"

        # Create unique session name
        session_name = f"vllm_{port}_{gpu_ids}".replace(",", "")
        print(f"{command}")
        if dry_run:
            return
        # Kill existing session if it exists
        subprocess.run(f"tmux kill-session -t {session_name} 2>/dev/null", shell=True)
        subprocess.run(f'tmux new-session -d -s {session_name} "{command}"', shell=True)
        print(
            f"Started vLLM server on GPUs {gpu_ids} at port {port} in tmux session '{session_name}'"
        )
    except Exception as e:
        print(f"Error starting server: {e}")


def stop_server(gpus=None, grep_pattern=None):
    try:
        processes = scan_vllm_process()
        killed = False
        for process in processes:
            should_kill = False
            if gpus and any(gpu in process["gpu_ids"] for gpu in gpus):
                should_kill = True
            if (
                grep_pattern
                and process.get("model_name")
                and grep_pattern in process["model_name"]
            ):
                should_kill = True

            if should_kill:
                try:
                    os.kill(process["pid"], signal.SIGTERM)
                    print(
                        f"Killed process {process['pid']} running {process.get('model_name', 'N/A')}"
                    )
                    killed = True
                    # Kill associated tmux session
                    for gpu in process["gpu_ids"]:
                        subprocess.run(
                            f"tmux ls | grep \"vllm.*{gpu}\" | cut -d ':' -f1 | xargs -I {{}} tmux kill-session -t {{}}",
                            shell=True,
                        )
                except ProcessLookupError:
                    continue

        if not killed:
            if gpus:
                print(f"No active vLLM processes found for GPUs {gpus}")
            if grep_pattern:
                print(
                    f"No active vLLM processes found matching pattern '{grep_pattern}'"
                )
    except Exception as e:
        print(f"Error stopping server: {e}")


def list_servers():
    try:
        processes = scan_vllm_process()
        if not processes:
            print("No active vLLM servers found")
            return

        print("\nActive vLLM Servers:")
        print("-" * 70)
        print(f"{'PID':<8} {'Port':<8} {'Model Name':<50}")
        print("-" * 70)
        for proc in processes:
            try:
                print(
                    f"{proc['pid']:<8} {proc['port']:<8} {proc['model_name'] or 'N/A':<50}"
                )
            except:
                continue
    except Exception as e:
        print(f"Error listing servers: {e}")


def print_examples():
    examples = """
Usage Examples:
--------------
1. Start a server with Qwen 32B model on GPUs 0-3:
    vllm-server start 32 --gpus 0123

2. Start with custom model from HuggingFace:
    vllm-server start meta-llama/Llama-2-70b-chat-hf --gpus 0123

3. Start with LoRA adapters:
    vllm-server start 7B --gpus 0123 --lora-modules sql-lora=/path/to/sql/lora alpaca=/path/to/alpaca/lora

4. List running servers:
    vllm-server ls

5. Stop servers on specific GPUs:
    vllm-server stop_at --gpus 0123

6. Stop servers by model name pattern:
    vllm-server stop --grep 32B

7. Dry run to check command:
    vllm-server start 32 --gpus 0123 --dry-run
"""
    print(examples)


def main():
    if not args.command:
        parser.print_help()
        print("\n")  # Add a newline for better readability
        print_examples()
        return

    if args.command == "start":
        list_gpus = list(args.gpus)
        # Parse LoRA modules if provided
        lora_modules = None
        if args.lora_modules:
            lora_modules = dict(module.split("=") for module in args.lora_modules)

        if "," in args.gpus:
            list_gpus = args.gpus.split(",")
        else:
            list_gpus = [args.gpus]
        for i, gpus in enumerate(list_gpus):
            start_server(
                args.model_size,
                gpus,
                args.port + i,
                lora_modules,
                args.dry_run,
                gpu_util=args.gpu_util,
                max_model_len=args.max_tokens,
                extra_args=args.extra,
            )
    elif args.command == "stop_at":
        list_gpus = list(args.gpus)
        stop_server(gpus=list_gpus)
    elif args.command == "stop":
        stop_server(grep_pattern=args.grep)

    elif args.command == "ls":
        list_servers()
    else:
        parser.print_help()
        print("\n")  # Add a newline for better readability
        print_examples()


if __name__ == "__main__":
    main()
    speedy_timer.print_task_table()
