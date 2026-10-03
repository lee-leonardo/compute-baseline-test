"""Platform selection and optional NVIDIA telemetry, outside the harness."""

import torch

from .protocol import MetricSpec, Observation

GPU_SCOPE = "whole selected device, including other processes"


class UnavailableGPU:
    specs = {
        "gpu_utilization": MetricSpec("percent", GPU_SCOPE, "unsupported"),
        "gpu_device_used": MetricSpec("bytes", GPU_SCOPE, "unsupported"),
    }

    def __init__(self, reason="No portable utilization counter for the selected backend"):
        self.reason = reason

    def open(self):
        pass

    def sample(self):
        return {name: Observation(None, self.reason) for name in self.specs}

    def close(self):
        pass


class NvidiaCollector(UnavailableGPU):
    specs = {
        "gpu_utilization": MetricSpec("percent", GPU_SCOPE, "NVML"),
        "gpu_device_used": MetricSpec("bytes", GPU_SCOPE, "NVML"),
        "gpu_power": MetricSpec("watts", "selected GPU board", "NVML"),
        "gpu_temperature": MetricSpec("degrees_celsius", GPU_SCOPE, "NVML"),
    }

    def __init__(self, uuid):
        super().__init__()
        self.uuid = uuid
        self.nvml = self.handle = None

    def open(self):
        initialized = False
        try:
            import pynvml

            pynvml.nvmlInit()
            initialized = True
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(self.uuid)
            self.nvml = pynvml
        except Exception as exc:
            self.reason = f"NVML unavailable: {type(exc).__name__}"
            if initialized:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

    def sample(self):
        if self.nvml is None:
            return super().sample()
        queries = {
            "gpu_utilization": lambda: self.nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu,
            "gpu_device_used": lambda: self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used,
            "gpu_power": lambda: self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000,
            "gpu_temperature": lambda: self.nvml.nvmlDeviceGetTemperature(
                self.handle, self.nvml.NVML_TEMPERATURE_GPU
            ),
        }
        result = {}
        for name, query in queries.items():
            try:
                result[name] = Observation(query())
            except Exception as exc:
                result[name] = Observation(None, f"NVML sample unavailable: {type(exc).__name__}")
        return result

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


def select_collector(args):
    if args.runtime == "torch" and args.device in ("auto", "cuda") and torch.cuda.is_available():
        try:
            uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
            return NvidiaCollector(uuid)
        except Exception as exc:
            return UnavailableGPU(f"GPU identity unavailable: {type(exc).__name__}")
    return UnavailableGPU()
