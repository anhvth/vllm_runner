import subprocess
import socket
from loguru import logger


def scan_available_gpus(util_threshold=0.9, mem_threshold=0.9):
    """Scan available GPUs on the machine with utilization and memory usage less than thresholds."""
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        lines = result.stdout.decode("utf-8").strip().split("\n")
        available_gpus = []
        for line in lines:
            gpu_id, util, mem_used, mem_total = line.split(", ")
            util = float(util.strip(" %")) / 100
            mem_used = float(mem_used.strip(" MiB"))
            mem_total = float(mem_total.strip(" MiB"))
            mem_util = mem_used / mem_total

            if util < util_threshold and mem_util < mem_threshold:
                available_gpus.append(gpu_id)

        logger.info(
            f"Available GPUs with utilization < {util_threshold} and memory < {mem_threshold}: {available_gpus}"
        )
        return available_gpus
    except subprocess.CalledProcessError as e:
        logger.error(f"Error scanning available GPUs: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error scanning available GPUs: {e}")
        return []


def scanfree_port(start=2800, ping_range=30):
    """Scan for free ports in the range [start, start+ping_range]."""
    free_ports = []
    for port in range(start, start + ping_range + 1):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", port))
            free_ports.append(port)
        except OSError:
            pass
        finally:
            sock.close()

    logger.info(f"Free ports in range [{start}, {start+ping_range}]: {free_ports}")
    return free_ports


def free_gpu(gpu_id, do_kill=False):
    """Scan and optionally kill processes using specified GPU."""
    try:
        result = subprocess.run(
            f"nvidia-smi -i {gpu_id} --query-compute-apps=pid --format=csv,noheader",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        pids = result.stdout.decode("utf-8").strip().split("\n")
        pids = [int(pid) for pid in pids if pid.strip()]

        if not pids:
            logger.info(f"No processes found using GPU {gpu_id}")
            return []

        logger.info(f"Found processes using GPU {gpu_id}: {pids}")

        if do_kill:
            for pid in pids:
                try:
                    subprocess.run(f"kill -9 {pid}", shell=True, check=True)
                    logger.info(f"Killed process {pid}")
                except subprocess.CalledProcessError:
                    logger.error(f"Failed to kill process {pid}")

        return pids
    except subprocess.CalledProcessError as e:
        logger.error(f"Error scanning GPU processes: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []


def free_vllm(at_port=None, model_name="7B", do_kill=False):
    """Find and optionally kill vLLM processes based on port or model name."""
    try:
        # Get all Python processes that might be vLLM
        result = subprocess.run(
            "ps aux | grep python | grep vllm",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        lines = result.stdout.decode("utf-8").strip().split("\n")
        pids = []

        for line in lines:
            if "grep" in line:  # Skip grep process itself
                continue

            parts = line.split()
            pid = int(parts[1])
            cmdline = " ".join(parts[10:])

            # Check if process matches criteria
            if at_port and f"--port {at_port}" in cmdline:
                pids.append(pid)
            elif model_name and model_name in cmdline:
                pids.append(pid)

        if not pids:
            logger.info(
                f"No vLLM processes found matching port={at_port} or model={model_name}"
            )
            return []

        logger.info(f"Found vLLM processes: {pids}")

        # Only kill if do_kill is True
        if do_kill:
            for pid in pids:
                try:
                    subprocess.run(f"kill -9 {pid}", shell=True, check=True)
                    logger.info(f"Killed vLLM process {pid}")
                except subprocess.CalledProcessError:
                    logger.error(f"Failed to kill process {pid}")

        return pids

    except subprocess.CalledProcessError as e:
        logger.error(f"Error scanning vLLM processes: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return []


if __name__ == "__main__":
    scan_available_gpus()
    scanfree_port()
    free_gpu(7, do_kill=False)
