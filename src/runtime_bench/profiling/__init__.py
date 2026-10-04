"""Runtime-independent telemetry collection boundary."""

from .profiler import ResourceSampler as Profiler
from .protocol import Collector, MetricSpec, Observation

__all__ = ["Profiler", "Collector", "MetricSpec", "Observation"]
