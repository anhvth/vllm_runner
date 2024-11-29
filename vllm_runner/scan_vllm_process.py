import os
import subprocess
from loguru import logger
import psutil
from dotenv import load_dotenv
from speedy_utils import speedy_timer

load_dotenv("dotenv.pub")

grep_name = os.environ.get("VLLM_PATH", "vllm.entrypoints.openai.api_server")


def scan_vllm_process() -> list[dict]:
    speedy_timer.tick()
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"])
            if grep_name and grep_name in cmdline:
                # Parse command line arguments
                args = proc.info["cmdline"]
                process_info = {
                    "pid": proc.info["pid"],
                    "port": None,
                    "model_name": None,
                }

                # Extract port and served-model-name from arguments
                for i, arg in enumerate(args):
                    if arg == "--port" and i + 1 < len(args):
                        process_info["port"] = args[i + 1]
                    elif arg == "--served-model-name" and i + 1 < len(args):
                        process_info["model_name"] = args[i + 1]

                processes.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except Exception as e:
            logger.error(f"Error: {e}")
    speedy_timer.update_task("scan_vllm_process")
    return processes


if __name__ == "__main__":
    print(scan_vllm_process())
