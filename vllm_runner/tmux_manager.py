import argparse
import os
import subprocess
import sys
import threading
from loguru import logger
from vllm_runner.scan_gpus import (
    kill_existing_vllm_processes,
    scan_available_gpus,
    scanfree_port,
)
from dotenv import load_dotenv
load_dotenv('dotenv.pub')

VLLM_PATH = os.environ.get("VLLM_PATH", "vllm")
FREE_PORT_LOCK = threading.Lock()
FREE_GPU_LOCK = threading.Lock()
FREE_PORT = scanfree_port()
GPUS = scan_available_gpus()


def select_port():
    with FREE_PORT_LOCK:
        return FREE_PORT.pop(0)


def get_gpu_ids(gpus, tp):
    with FREE_GPU_LOCK:
        gpu_ids = gpus[:tp]
        del gpus[:tp]
    return gpu_ids


def create_tmux_window(session_name, window_name, command):
    try:
        tmux_command = f'tmux new-window -t {session_name} -n {window_name} "{command}"'
        subprocess.run(tmux_command, shell=True, check=True)
        logger.info(
            f"Created window '{window_name}' in session '{session_name}'\n{command}"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating tmux window '{window_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating tmux window '{window_name}': {e}")


def create_tmux_session(session_name, initial_command, initial_window_name):
    try:
        tmux_command = f'tmux new-session -d -s {session_name} -n {initial_window_name} "{initial_command}"'
        subprocess.run(tmux_command, shell=True, check=True)
        logger.info(
            f"Created new tmux session '{session_name}' with initial window '{initial_window_name}'"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating tmux session '{session_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating tmux session '{session_name}': {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Launch vllm servers in tmux windows.")
    parser.add_argument(
        "--gpu-ids",
        "-g",
        type=str,
        default=None,
        help='Comma-separated list of GPU IDs (default: "0,1,3,4,5,6")',
    )
    parser.add_argument(
        "-tp", type=int, default=1, help="Number of tensor parallel GPUs (default: 1)"
    )
    parser.add_argument(
        "--max-length",
        "-l",
        type=int,
        default=30000,
        help="Maximum model length (default: 30000)",
    )
    parser.add_argument(
        "--model-path", "-m", default=None, help="Path to the model (default: None)"
    )
    parser.add_argument(
        "--port-start",
        "-p",
        type=int,
        default=None,
        help="Starting port number (default: auto-select)",
    )
    parser.add_argument(
        "--model-size",
        "-s",
        type=str,
        default="32B",
        help='Model size (default: "32B")',
    )
    parser.add_argument(
        "-f",
        "--force-kill",
        action="store_true",
        help="Terminate all current vllm processes and tmux session before starting new ones",
    )
    parser.add_argument(
        "--gpu-util",
        default=0.95,
        type=float,
        help="GPU utilization threshold (default: 0.95)",
    )
    return parser


def main(args):
    session_name = "vllm_runner"

    if args.force_kill:
        kill_existing_vllm_processes()
        try:
            subprocess.run(
                f"tmux kill-session -t {session_name}", shell=True, check=True
            )
            logger.info(f"Killed existing tmux session: {session_name}")
        except subprocess.CalledProcessError:
            logger.warning(f"No existing tmux session named '{session_name}' to kill.")

    session_exists = (
        subprocess.run(
            f"tmux has-session -t {session_name}", shell=True, check=False
        ).returncode
        == 0
    )

    if session_exists and not args.force_kill:
        logger.info(f"Tmux session '{session_name}' exists. Appending new windows.")

    tp = args.tp
    commands = []

    def get_command():
        propose_port = select_port()
        _gpu_ids = get_gpu_ids(GPUS, tp)
        logger.info(f"Selected GPUs: {_gpu_ids}")
        gpu_str = ",".join(list(_gpu_ids))
        model = args.model_path or f"Qwen/Qwen2.5-{args.model_size}-Instruct-AWQ"
        served_model_name = (
            os.path.basename(args.model_path)
            if args.model_path
            else model.split("/")[-1]
        )
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"{VLLM_PATH} serve {model} "
            f"--tensor-parallel-size {args.tp} "
            f"--gpu-memory-utilization {args.gpu_util} "
            f"--trust-remote-code "
            f"--dtype half "
            f"--enforce-eager "
            f"--max-model-len {args.max_length} "
            f"--swap-space 16 "
            f"--port {propose_port} "
            f"--served-model-name {served_model_name} "
            f"--enable-prefix-caching "
        )
        window_name = f"vllm_{propose_port}"
        commands.append((window_name, command))

    get_command()
    logger.info(f"Commands: {commands}")

    if not commands:
        logger.error("No commands to execute. Exiting.")
        sys.exit(1)

    if not session_exists:
        first_window_name, first_command = commands.pop(0)
        create_tmux_session(session_name, first_command, first_window_name)

    for window_name, command in commands:
        create_tmux_window(session_name, window_name, command)
        logger.success(f"Created window '{window_name}' with command: {command}")

    logger.info(f"To attach to the tmux session, use: tmux attach -t {session_name}")


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
