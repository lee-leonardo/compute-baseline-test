import platform
import subprocess

import psutil
import torch


def command(*args):
    try:
        return subprocess.check_output(
            args, text=True, timeout=5, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def identify(device):
    cpu = platform.processor() or "unknown"
    if platform.system() == "Darwin":
        cpu = command("sysctl", "-n", "machdep.cpu.brand_string")
    elif platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    vendor = next((v for v in ("Apple", "Intel", "AMD") if v.lower() in cpu.lower()), "unknown")
    gpu, gpu_vendor, memory, unified = "none selected", "none", None, "unknown"
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        gpu, gpu_vendor, memory, unified = p.name, "NVIDIA", p.total_memory, False
    elif device.type == "mps":
        gpu, gpu_vendor, unified = cpu + " integrated GPU", "Apple", True
    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "cpu_vendor": vendor,
        "cpu_logical_cores": psutil.cpu_count(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu": gpu,
        "gpu_vendor": gpu_vendor,
        "gpu_memory_bytes": memory,
        "unified_memory": unified,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "cuda_driver": command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
        if device.type == "cuda"
        else "not applicable",
        "cuda_capability": torch.cuda.get_device_capability(device)
        if device.type == "cuda"
        else None,
    }


def select_device(request):
    if request == "auto":
        request = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested but unavailable; install --extra cuda and check the driver"
        )
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    return torch.device(request)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
