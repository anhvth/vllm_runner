
import subprocess

def get_gpu_info():
    try:
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"
        ])
        
        gpus = []
        for line in output.decode('utf-8').strip().split('\n'):
            index, name, temp, util, mem_used, mem_total, power = line.split(',')
            gpus.append({
                'index': int(index),
                'name': name.strip(),
                'temperature': float(temp),
                'utilization': float(util),
                'memory_used': float(mem_used),
                'memory_total': float(mem_total),
                'power_draw': float(power)
            })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

if __name__ == "__main__":
    gpu_info = get_gpu_info()
    for gpu in gpu_info:
        print(gpu)
