import argparse
import json
import os
import socket
import subprocess
import sys
import time

from loguru import logger

from vllm_runner.scan_gpus import scan_available_gpus, scanfree_port
import threading

FREE_PORT_LOCK = threading.Lock()
FREE_PORT = scanfree_port()
GPUS = scan_available_gpus()

def select_port():
    with FREE_PORT_LOCK:
        port = FREE_PORT.pop(0)
    return port


def kill_existing_vllm_processes():
    """Finds and kills all running vllm processes, then waits until they are terminated."""
    try:
        # Kill all vllm processes
        subprocess.run("pgrep -f vllm | xargs -r kill -9", shell=True, check=False)
        logger.info("Attempting to kill all running vllm processes...")

        # Wait in a loop until all vllm processes are completely terminated
        while True:
            result = subprocess.run(
                "pgrep -f vllm", shell=True, check=False, stdout=subprocess.PIPE
            )
            if result.returncode != 0:  # No vllm processes found
                logger.info("All vllm processes have been successfully terminated.")
                break
            else:
                logger.warning("Waiting for vllm processes to terminate...")
                time.sleep(1)  # Wait for 1 second before checking again

    except Exception as e:
        logger.error(f"Error while killing vllm processes: {e}")


def parse_gpu_ids(gpu_ids_str):
    """Parses a string of GPU IDs in the format '0,1,3,4,5,6' to a list of strings."""
    try:
        # Split by commas and strip whitespace
        gpu_ids = [gpu.strip() for gpu in gpu_ids_str.split(",") if gpu.strip()]
        # Validate that each GPU ID is an integer
        for gpu in gpu_ids:
            if not gpu.isdigit():
                raise ValueError(f"Invalid GPU ID: {gpu}")
        logger.debug(f"Parsed GPU IDs: {gpu_ids} from string: {gpu_ids_str}")
        return gpu_ids
    except ValueError as e:
        logger.error("GPU IDs must be a comma-separated list of integers.")
        raise argparse.ArgumentTypeError(
            "GPU IDs must be a comma-separated list of integers."
        ) from e


def create_tmux_window(session_name, window_name, command):
    """Create a new tmux window within the specified session and run the given command."""
    try:
        # Create a new window in the session and run the command
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
    """Create a new tmux session with the first window running the initial command."""
    try:
        # Create a new detached tmux session with the initial window running the command
        tmux_command = f'tmux new-session -d -s {session_name} -n {initial_window_name} "{initial_command}"'
        subprocess.run(tmux_command, shell=True, check=True)
        logger.info(
            f"Created new tmux session '{session_name}' with initial window '{initial_window_name}'"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating tmux session '{session_name}': {e}")
    except Exception as e:
        logger.error(f"Unexpected error creating tmux session '{session_name}': {e}")


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Launch vllm servers in tmux windows. Example usage: python tools/vllm_tmux_manager.py -s 72B -g 1234 -l 15000 -p 2804"
    )
    parser.add_argument(
        "--gpu-ids",
        "-g",
        type=str,
        default=None
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

    args = parser.parse_args()
    # Define tmux session name
    session_name = "vllm_runner"

    # Kill existing vllm processes and tmux session if the -f flag is set
    if args.force_kill:
        kill_existing_vllm_processes()
        try:
            subprocess.run(
                f"tmux kill-session -t {session_name}", shell=True, check=True
            )
            logger.info(f"Killed existing tmux session: {session_name}")
        except subprocess.CalledProcessError:
            logger.warning(f"No existing tmux session named '{session_name}' to kill.")

    # Check if tmux session exists
    session_exists = False
    try:
        result = subprocess.run(
            f"tmux has-session -t {session_name}", shell=True, check=False
        )
        if result.returncode == 0:
            session_exists = True
            if not args.force_kill:
                logger.info(
                    f"Tmux session '{session_name}' exists. Appending new windows."
                )
        else:
            session_exists = False
    except Exception as e:
        logger.error(f"Error checking tmux session '{session_name}': {e}")
        sys.exit(1)

    # Parse GPU IDs
    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids)
    except argparse.ArgumentTypeError as e:
        logger.error(e)
        sys.exit(1)

    max_length = args.max_length
    model_size = args.model_size

    # Loop over each GPU ID to create commands
    commands = []
    propose_port = args.port_start
    # for i, _gpu_ids in enumerate(gpu_ids):
    tp = args.tp
    for i in range(0, len(gpu_ids), tp):
        _gpu_ids = gpu_ids[i : i + tp]
        propose_port = select_port(propose_port)
        if len(_gpu_ids) != tp:
            logger.error(
                f"Number of GPUs ({len(_gpu_ids)}) must be a multiple of tensor parallel size ({tp})."
            )
            break
        gpu_str = ",".join(list(_gpu_ids))
        model = (
            args.model_path or f"Qwen/Qwen2.5-{model_size}-Instruct-AWQ"
        )  
        # Command to be executed in the tmux window
        if args.model_path:
            if "/saves/":  # hardcoded for now, need to change this later
                served_model_name = args.model_path.split("/saves/")[-1]
            else:
                served_model_name = os.path.basename(args.model_path)
        else:
            served_model_name = model.split("/")[-1]
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"vllm serve {model} "
            f"--tensor-parallel-size {args.tp} "
            f"--gpu-memory-utilization {args.gpu_util} "
            f"--trust-remote-code "
            f"--dtype half "
            f"--enforce-eager "
            f"--max-model-len {max_length} "
            f"--swap-space 16 "
            f"--port {propose_port} "
            f"--served-model-name {served_model_name} "
            f"--enable-prefix-caching "
        )

        print(f"{command}")
        # Window name will be based on the port
        window_name = f"vllm_{propose_port}"
        propose_port += 1

        commands.append((window_name, command))

    if not commands:
        logger.error("No GPU IDs provided. Exiting.")
        sys.exit(1)

    if not session_exists:
        # Create the tmux session with the first window
        first_window_name, first_command = commands.pop(0)
        create_tmux_session(session_name, first_command, first_window_name)

        # Create additional windows
        for window_name, command in commands:
            create_tmux_window(session_name, window_name, command)
    else:
        # Append new windows to existing session
        for window_name, command in commands:
            create_tmux_window(session_name, window_name, command)

    logger.info("All tmux windows have been created successfully.")
    logger.info(f"To attach to the tmux session, use: tmux attach -t {session_name}")


if __name__ == "__main__":
    main()
