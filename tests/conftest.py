import pytest


def pytest_addoption(parser):
    parser.addoption("--run-mlx", action="store_true", help="Run native Apple MLX GPU checks")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-mlx"):
        skip = pytest.mark.skip(reason="Native MLX checks require --run-mlx on Apple Silicon")
        for item in items:
            if "mlx" in item.keywords:
                item.add_marker(skip)
