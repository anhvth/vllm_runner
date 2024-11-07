import subprocess
import time

import gradio as gr

from vllm_runner.scan_gpus import scan_available_gpus, scanfree_port
from vllm_runner.tmux_manager import main, parse_arguments


def get_gpu_stats():
    """Get formatted GPU statistics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=id,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        lines = result.stdout.decode("utf-8").strip().split("\n")

        formatted_output = "GPU Statistics:\n"
        formatted_output += "-" * 100 + "\n"
        formatted_output += f"{'GPU ID':<8}{'Name':<20}{'Temp':<8}{'GPU%':<8}{'Memory':<20}{'Power':<12}\n"
        formatted_output += "-" * 100 + "\n"

        for line in lines:
            gpu_id, name, temp, util, mem_used, mem_total, power = [
                x.strip() for x in line.split(",")
            ]
            memory = f"{mem_used}/{mem_total}"
            formatted_output += f"{gpu_id:<8}{name[:18]:<20}{temp}C{' ':<4}{util:<8}{memory:<20}{power:<12}\n"

        return formatted_output
    except Exception as e:
        return f"Error getting GPU stats: {e}"


def run_tmux_manager_gradio(
    gpu_ids,
    tp,
    max_length,
    model_path,
    port_start,
    model_size,
    force_kill,
    gpu_util,
    new_param,
):
    parser = parse_arguments()
    # Scan for available GPUs
    available_gpus = scan_available_gpus(util_threshold=gpu_util, mem_threshold=0.9)
    if not available_gpus:
        return "No available GPUs found!"

    # Parse requested GPU count
    requested_count = len(gpu_ids.strip().split(",")) if gpu_ids else 1

    # Check if we have enough GPUs
    if len(available_gpus) < requested_count:
        return f"Not enough available GPUs! Requested {requested_count}, but only {len(available_gpus)} available"

    # Get the required number of GPUs
    selected_gpus = available_gpus[:requested_count]
    gpu_ids = ",".join(selected_gpus)

    # Get available ports
    available_ports = scanfree_port(start=int(port_start) if port_start else 2800)
    if not available_ports:
        return "No available ports found!"

    port_start = available_ports[0]

    # Prepare arguments
    parser = parse_arguments()
    args_list = [
        "--gpu-ids",
        gpu_ids,
        "-tp",
        str(int(tp)),
        "--max-length",
        str(int(max_length)),
        "--port-start",
        str(port_start),
        "--model-size",
        model_size,
        "--gpu-util",
        str(gpu_util),
        "--new-param",
        new_param,
    ]

    if model_path:
        args_list.extend(["--model-path", model_path])
    if force_kill:
        args_list.append("--force-kill")

    args = parser.parse_args(args_list)
    main(args)
    return f"Tmux manager executed using GPUs {gpu_ids} starting at port {port_start}"


def refresh_stats():
    return get_gpu_stats()


def gpu_monitor_live():
    """Generator function to yield GPU stats periodically."""
    while True:
        yield get_gpu_stats()
        time.sleep(5)  # Update every 5 seconds


with gr.Blocks(title="Tmux Manager with GPU Monitor") as iface:
    with gr.Row():
        with gr.Column(scale=2):
            gpu_ids = gr.Textbox(label="GPU IDs (comma-separated)", value="0,1")
            tp = gr.Number(label="Tensor Parallel Size", value=1)
            max_length = gr.Number(label="Max Model Length", value=30000)
            model_path = gr.Textbox(label="Model Path", value="")
            port_start = gr.Number(label="Port Start", value=2800)
            model_size = gr.Textbox(label="Model Size", value="32B")
            force_kill = gr.Checkbox(label="Force Kill Existing Processes", value=False)
            gpu_util = gr.Slider(
                label="GPU Utilization Threshold", minimum=0.0, maximum=1.0, value=0.95
            )
            new_param = gr.Textbox(label="New Parameter", value="default")
            submit_btn = gr.Button("Run Tmux Manager")
            output = gr.Textbox(label="Output")

        with gr.Column(scale=1):
            with gr.Live(gpu_monitor_live) as live:
                gpu_monitor = gr.Textbox(
                    label="GPU Monitor",
                    value=get_gpu_stats(),
                    lines=15,
                    max_lines=20,
                    interactive=False,
                )
            refresh_btn = gr.Button("Refresh GPU Stats")

    submit_btn.click(
        fn=run_tmux_manager_gradio,
        inputs=[
            gpu_ids,
            tp,
            max_length,
            model_path,
            port_start,
            model_size,
            force_kill,
            gpu_util,
            new_param,
        ],
        outputs=output,
    )

    refresh_btn.click(fn=refresh_stats, inputs=[], outputs=gpu_monitor)

    iface.launch()

if __name__ == "__main__":
    iface.launch()
