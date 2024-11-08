import argparse
import subprocess
import os

from speedy_utils import fprint
from vllm_runner.scan_gpus import scan_available_gpus, free_gpu
from vllm_runner.scan_vllm_process import scan_vllm_process
from vllm_runner.app import get_required_tp, scanfree_port

def start_server(model_size, gpus, port_start):
    try:
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
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model Qwen2.5-{model_size}-Instruct-AWQ "
            f"--tensor-parallel-size {tp} "
            f"--port {port} "
            f"--host 0.0.0.0"
        )
        # Create unique session name
        session_name = f"vllm_{port}_{gpu_ids}"
        # Kill existing session if it exists
        subprocess.run(f'tmux kill-session -t {session_name} 2>/dev/null', shell=True)
        subprocess.run(f'tmux new-session -d -s {session_name} "{command}"', shell=True)
        print(f"Started vLLM server on GPUs {gpu_ids} at port {port} in tmux session '{session_name}'")
    except Exception as e:
        print(f"Error starting server: {e}")

def stop_server(gpus):
    try:
        # Find all vLLM processes associated with the specified GPUs
        processes = scan_vllm_process()
        killed = False
        for process in processes:
            if any(gpu in process['gpu_ids'] for gpu in gpus):
                try:
                    os.kill(process['pid'], signal.SIGTERM)
                    print(f"Killed process {process['pid']} on GPUs {process['gpu_ids']}")
                    killed = True
                except ProcessLookupError:
                    continue
        
        # Kill associated tmux sessions
        for gpu in gpus:
            subprocess.run(f'tmux ls | grep "vllm.*{gpu}" | cut -d \':\' -f1 | xargs -I {{}} tmux kill-session -t {{}}', shell=True)
        
        if not killed:
            print(f"No active vLLM processes found for GPUs {gpus}")
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
                print(f"{proc['pid']:<8} {proc['port']:<8} {proc['model_name'] or 'N/A':<50}")
            except:
                continue
    except Exception as e:
        print(f"Error listing servers: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage vLLM servers")
    subparsers = parser.add_subparsers(dest='command')

    start_parser = subparsers.add_parser('start', help='Start a vLLM server')
    start_parser.add_argument('model_size', type=str, help='Model size (e.g., 72B)')
    start_parser.add_argument('--gpus', type=str, required=True, help='GPUs to use (e.g., 0123)')
    start_parser.add_argument('--port', type=int, default=2800, help='Starting port number')

    stop_parser = subparsers.add_parser('stop_at', help='Stop vLLM servers at specified GPUs')
    stop_parser.add_argument('--gpus', type=str, required=True, help='GPUs to free (e.g., 0123)')
    
    # Add new ls subparser
    subparsers.add_parser('ls', help='List all running vLLM servers')

    args = parser.parse_args()

    if args.command == 'start':
        gpus = list(args.gpus)
        start_server(args.model_size, gpus, args.port)
    elif args.command == 'stop_at':
        gpus = list(args.gpus)
        stop_server(gpus)
    elif args.command == 'ls':
        list_servers()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()