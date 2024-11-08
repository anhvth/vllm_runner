
import argparse
import subprocess
import os
from vllm_runner.scan_gpus import scan_available_gpus, free_gpu
from vllm_runner.scan_vllm_process import scan_vllm_process
from vllm_runner.app import get_required_tp, scanfree_port

def start_server(model_size, gpus, port_start):
    # Ensure GPUs are available
    available_gpus = scan_available_gpus()
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
    tp = get_required_tp(model_size)
    command = (
        f"CUDA_VISIBLE_DEVICES={gpu_ids} "
        f"vllm serve Qwen2.5-{model_size}-Instruct-AWQ "
        f"--tensor-parallel-size {tp} "
        f"--port {port} "
    )
    # Start the server in a new tmux session
    session_name = f"vllm_{port}"
    subprocess.run(f'tmux new-session -d -s {session_name} "{command}"', shell=True)
    print(f"Started vLLM server on GPUs {gpu_ids} at port {port} in tmux session '{session_name}'")

def stop_server(gpus):
    # Find all vLLM processes associated with the specified GPUs
    processes = scan_vllm_process()
    pids_to_kill = []
    for process in processes:
        if any(gpu in process['gpu_ids'] for gpu in gpus):
            pids_to_kill.append(process['pid'])
    if not pids_to_kill:
        print(f"No vLLM processes found for GPUs {gpus}")
        return
    # Kill the processes
    for pid in pids_to_kill:
        subprocess.run(f"kill -9 {pid}", shell=True)
        print(f"Killed process {pid}")
    # Optionally, kill tmux sessions associated with these processes
    # ...

def main():
    parser = argparse.ArgumentParser(description="Manage vLLM servers")
    subparsers = parser.add_subparsers(dest='command')

    start_parser = subparsers.add_parser('start', help='Start a vLLM server')
    start_parser.add_argument('model_size', type=str, help='Model size (e.g., 72B)')
    start_parser.add_argument('--gpus', type=str, required=True, help='GPUs to use (e.g., 0123)')
    start_parser.add_argument('--port', type=int, default=2800, help='Starting port number')

    stop_parser = subparsers.add_parser('stop_at', help='Stop vLLM servers at specified GPUs')
    stop_parser.add_argument('--gpus', type=str, required=True, help='GPUs to free (e.g., 0123)')

    args = parser.parse_args()

    if args.command == 'start':
        gpus = list(args.gpus)
        start_server(args.model_size, gpus, args.port)
    elif args.command == 'stop_at':
        gpus = list(args.gpus)
        stop_server(gpus)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()