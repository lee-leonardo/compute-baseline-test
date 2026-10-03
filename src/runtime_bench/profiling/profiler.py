"""Small in-process sampler; unavailable GPU counters remain explicitly unavailable."""

import threading

import psutil
from .collectors import select_collector


class ResourceSampler:
    def __init__(self, args, interval=0.25, collector=None):
        self.interval = interval
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.thread = None
        self.rss_peak = 0
        self.cpu_values = []
        self.gpu_values = []
        self.device_memory_peak = None
        self.available_min = psutil.virtual_memory().available
        self.swap_start = psutil.swap_memory().used
        self.samples = 0
        self.collector = collector or select_collector(args)
        self.metric_values = {}
        self.metric_errors = {}
        self.metric_failures = {}
        self.collector.open()

    def sample(self):
        self.rss_peak = max(self.rss_peak, self.process.memory_info().rss)
        self.available_min = min(self.available_min, psutil.virtual_memory().available)
        self.cpu_values.append(self.process.cpu_percent())
        self.samples += 1
        for name, observation in self.collector.sample().items():
            if observation.value is not None:
                self.metric_values.setdefault(name, []).append(observation.value)
            if observation.reason:
                self.metric_errors[name] = observation.reason
                self.metric_failures[name] = self.metric_failures.get(name, 0) + 1
        self.gpu_values = self.metric_values.get("gpu_utilization", [])
        memory = self.metric_values.get("gpu_device_used", [])
        self.device_memory_peak = max(memory) if memory else None
        self.gpu_reason = self.metric_errors.get("gpu_utilization")

    def loop(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.process.cpu_percent()  # Prime the counter; do not include an initial zero.
        self.sample()
        self.cpu_values.clear()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()
        self.collector.close()

    def summary(self):
        return {
            "telemetry_protocol": "telemetry-v1",
            "collection_window": "whole job, including setup, warmup and evaluation",
            "metrics": self.metrics(),
            "sample_interval_seconds": self.interval,
            "samples": self.samples,
            "process_rss_sampled_peak_bytes": self.rss_peak,
            "process_cpu_mean_percent": sum(self.cpu_values) / len(self.cpu_values)
            if self.cpu_values
            else None,
            "process_cpu_peak_percent": max(self.cpu_values) if self.cpu_values else None,
            "gpu_utilization_mean_percent": sum(self.gpu_values) / len(self.gpu_values)
            if self.gpu_values
            else None,
            "gpu_utilization_peak_percent": max(self.gpu_values) if self.gpu_values else None,
            "gpu_device_used_sampled_peak_bytes": self.device_memory_peak,
            "gpu_utilization_scope": "whole selected device, including other processes",
            "gpu_utilization_unavailable_reason": self.gpu_reason,
            "host_available_ram_min_bytes": self.available_min,
            "host_swap_used_change_bytes": psutil.swap_memory().used - self.swap_start,
        }

    def metrics(self):
        result = {}
        for name, spec in self.collector.specs.items():
            values = self.metric_values.get(name, [])
            result[name] = {
                "unit": spec.unit,
                "scope": spec.scope,
                "source": spec.source,
                "status": "partial"
                if values and self.metric_errors.get(name)
                else "available"
                if values
                else "unavailable",
                "reason": self.metric_errors.get(name),
                "samples": len(values),
                "failed_samples": self.metric_failures.get(name, 0),
                "mean": sum(values) / len(values) if values else None,
                "sampled_peak": max(values) if values else None,
            }
        host_metrics = {
            "process_rss": ("bytes", "process resident memory", [self.rss_peak], "sampled_peak"),
            "process_cpu": (
                "percent",
                "process; 100 percent equals one CPU core",
                self.cpu_values,
                None,
            ),
            "host_available_ram": ("bytes", "whole host", [self.available_min], "sampled_min"),
        }
        for name, (unit, scope, values, aggregate) in host_metrics.items():
            metric = {
                "unit": unit,
                "scope": scope,
                "source": "psutil",
                "status": "available" if values else "unavailable",
                "reason": None,
                "samples": self.samples if aggregate else len(values),
                "failed_samples": 0,
            }
            if aggregate:
                metric[aggregate] = values[0]
            else:
                metric["mean"] = sum(values) / len(values) if values else None
                metric["sampled_peak"] = max(values) if values else None
            result[name] = metric
        return result
