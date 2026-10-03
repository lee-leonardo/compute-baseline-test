"""Collector contract; collectors never execute or change workloads."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MetricSpec:
    unit: str
    scope: str
    source: str


@dataclass(frozen=True)
class Observation:
    value: float | int | None
    reason: str | None = None


class Collector(Protocol):
    specs: dict[str, MetricSpec]

    def open(self) -> None: ...
    def sample(self) -> dict[str, Observation]: ...
    def close(self) -> None: ...
