import argparse
import os
import subprocess
import threading
from glob import glob

import gradio as gr
from dotenv import load_dotenv
from loguru import logger
from tabulate import tabulate

from vllm_runner.scan_gpus import (
    kill_existing_vllm_processes,
    scan_available_gpus,
    scanfree_port,
)
from vllm_runner.scan_vllm_process import scan_vllm_process

load_dotenv("dotenv.pub")

VLLM_PATH = os.environ.get("VLLM_PATH", "vllm")
assert os.path.exists(VLLM_PATH), f"vLLM path '{VLLM_PATH}' does not exist!"
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


def get_mock_gpu_stats():
    """Return a message when GPU stats are unavailable."""
    return "GPU Statistics unavailable:\nCould not connect to NVIDIA driver. Please ensure NVIDIA drivers are installed and working."


def get_gpu_stats():
    """Get GPU statistics using nvidia-smi command."""
    try:
        # Run nvidia-smi command with specific format
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Process and format the output
        data = []
        for line in result.stdout.strip().split("\n"):
            fields = line.strip().split(",")
            idx = fields[0].strip()
            util = fields[1].strip()
            mem_used = fields[2].strip()
            mem_total = fields[3].strip()
            mem_util = float(mem_used) / float(mem_total) * 100
            data.append([idx, f"{100 - mem_util:.1f}%"])

        headers = ["GPU Index", "FreeMem (%)"]
        formatted_output = tabulate(data, headers)
        return formatted_output
    except (subprocess.CalledProcessError, FileNotFoundError, ZeroDivisionError):
        return get_mock_gpu_stats()


def get_required_tp(model_size):
    mapping = {
        "7B": 1,
        "14B": 2,
        "32B": 4,
        "72B": 8,
    }
    return mapping.get(model_size, 1)  # Default to 1 if not found


def run_tmux_manager_gradio(
    max_length,
    tp,
    model_selection,
    port_start,
    gpu_util,
    start_tmux,
    
):
    session_name = "vllm_runner"

    # Determine if model_selection is a path or a predefined size
    if os.path.isdir(model_selection):
        model_path = model_selection
        model_size = None
    else:
        model_path = None
        model_size = model_selection

    # Ensure that a model is selected
    if not model_selection:
        return "Please select a model."

    # Ensure that both model_size and model_path are not provided
    if model_path and model_size:
        return "Error: Both model size and model path are provided. Please provide only one."

    # Automatically set tp based on model size
    if tp is None:
        tp = get_required_tp(model_size)

    # Automatically find available GPUs
    available_gpus = scan_available_gpus(mem_threshold=gpu_util)
    if not available_gpus:
        return "No available GPUs found!"

    # Check if enough GPUs are available
    if len(available_gpus) < tp:
        return f"Not enough GPUs available! Required: {tp}, Available: {len(available_gpus)}"

    # Select required number of GPUs
    selected_gpus = available_gpus[:tp]
    gpu_ids = ",".join(selected_gpus)

    # Automatically find available port
    available_ports = scanfree_port(start=int(port_start) if port_start else 2800)
    if not available_ports:
        return "No available ports found!"
    port_start = available_ports[0]

    # Prepare the command
    model = model_path or f"Qwen/Qwen2.5-{model_size}-Instruct-AWQ"
    served_model_name = (
        os.path.basename(model_path) if model_path else model.split("/")[-1]
    )
    command = (
        f"CUDA_VISIBLE_DEVICES={gpu_ids} "
        f"{VLLM_PATH} serve {model} "
        f"--tensor-parallel-size {tp} "
        f"--gpu-memory-utilization {gpu_util} "
        f"--trust-remote-code "
        f"--dtype half "
        f"--enforce-eager "
        f"--max-model-len {int(max_length)} "
        f"--swap-space 16 "
        f"--port {port_start} "
        f"--served-model-name {served_model_name} "
        f"--enable-prefix-caching "
    )

    window_name = f"vllm_{port_start}"

    session_exists = (
        subprocess.run(
            f"tmux has-session -t {session_name}", shell=True, check=False
        ).returncode
        == 0
    )

    # Build the intended commands
    if not session_exists:
        tmux_command = f'tmux new-session -d -s {session_name} -n {window_name} "{command}"'
    else:
        tmux_command = f'tmux new-window -t {session_name} -n {window_name} "{command}"'

    # Execute or display command based on start_tmux checkbox
    if start_tmux:
        if not session_exists:
            create_tmux_session(session_name, command, window_name)
        else:
            create_tmux_window(session_name, window_name, command)
        return f"Tmux manager executed using GPUs {gpu_ids} starting at port {port_start}"
    else:
        return f"Intended tmux command:\n{tmux_command}"


def refresh_stats():
    return get_gpu_stats()


def get_model_selections() -> list:
    """Get all model selections, combining model sizes and model paths."""
    # Predefined model sizes
    model_sizes = ["7B", "14B", "32B", "72B"]
    # Model paths
    glob_path = "../LLaMA-Factory/saves/*/*.awq/"
    model_paths = glob(glob_path)
    # Combine and return
    return model_sizes + model_paths


def format_process_name(process_list):
    """Format process names for display."""
    processes = {}
    for process in process_list:
        name = f"{process['model_name']} (Port {process['port']})"
        processes[name] = process['pid']
    return processes


def kill_selected_process(process_name):
    if not process_name:
        return "No process selected"
    processes = format_process_name(scan_vllm_process())
    pid = processes.get(process_name)
    if pid:
        try:
            subprocess.run(['kill', str(pid)], check=True)
            return f"Successfully killed process {process_name}"
        except subprocess.CalledProcessError:
            return f"Failed to kill process {process_name}"
    return "Process not found"


def refresh_process_list():
    processes = format_process_name(scan_vllm_process())
    return gr.update(choices=list(processes.keys()))


# Create Gradio interface
with gr.Blocks(title="vLLM Manager with GPU Monitor") as demo:
    with gr.Tabs():
        with gr.Tab("Tmux Manager"):
            with gr.Row():
                with gr.Column(scale=2):
                    max_length = gr.Number(label="Max Model Length", value=16000)
                    tp = gr.Dropdown(
                        label="Tensor Parallel Size", choices=[1, 2, 4, 8], value=4
                    )
                    model_selection = gr.Dropdown(
                        label="Model Selection",
                        choices=get_model_selections(),
                        value="32B",
                    )
                    port_start = gr.Number(label="Port Start", value=2800)
                    # force_kill = gr.Checkbox(
                    #     label="Force Kill Existing Processes", value=False
                    # )
                    gpu_util = gr.Slider(
                        label="GPU Utilization Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                    )
                    start_tmux = gr.Checkbox(label="Start Tmux", value=True)
                    submit_btn = gr.Button("Run Tmux Manager")
                    output = gr.Textbox(label="Output", lines=10)

            submit_btn.click(
                fn=run_tmux_manager_gradio,
                inputs=[
                    max_length,
                    tp,
                    model_selection,
                    port_start,
                    # force_kill,
                    gpu_util,
                    start_tmux,
                ],
                outputs=output,
            )

        with gr.Tab("GPU Monitor"):
            gpu_monitor = gr.Textbox(
                label="GPU Monitor",
                value=get_gpu_stats(),
                lines=15,
                max_lines=20,
                interactive=False,
            )
            refresh_btn = gr.Button("Refresh GPU Stats")

            refresh_btn.click(fn=refresh_stats, inputs=[], outputs=gpu_monitor)

            # Setup automatic refresh every 30 seconds
            demo.load(refresh_stats, outputs=gpu_monitor)

        with gr.Tab("vLLM Process Manager"):
            refresh_process_btn = gr.Button("Refresh Process List")
            
            process_dropdown = gr.Dropdown(
                label="Select vLLM Process to Kill",
                choices=list(format_process_name(scan_vllm_process()).keys()),
            )
            kill_btn = gr.Button("Kill Process")
            kill_output = gr.Textbox(label="Kill Output")

            kill_btn.click(
                fn=kill_selected_process,
                inputs=[process_dropdown],
                outputs=[kill_output],
            )

            refresh_process_btn.click(
                fn=refresh_process_list,
                inputs=[],
                outputs=[process_dropdown],
            )

if __name__ == "__main__":
    demo.launch(share=False)
