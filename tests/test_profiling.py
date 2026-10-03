from runtime_bench.cli import parser
from runtime_bench.profiling import MetricSpec, Observation, Profiler


def test_injected_collector_partial_counter_and_cleanup():
    class Fake:
        specs = {"gpu_utilization": MetricSpec("percent", "fake device", "test")}
        calls = 0
        closed = False

        def open(self):
            pass

        def sample(self):
            self.calls += 1
            return {
                "gpu_utilization": Observation(80)
                if self.calls == 1
                else Observation(None, "counter lost")
            }

        def close(self):
            self.closed = True

    collector = Fake()
    with Profiler(parser().parse_args(["smoke"]), collector=collector) as profiler:
        pass
    metric = profiler.summary()["metrics"]["gpu_utilization"]
    assert collector.closed
    assert metric["status"] == "partial"
    assert metric["mean"] == 80
    assert metric["failed_samples"] == 1
    assert metric["scope"] == "fake device"
