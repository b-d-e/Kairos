"""For CPU testing, simulates GPU memory usage."""

import subprocess

import pytest


class MockGPUProcess:
    """Mock GPU process for testing."""

    def __init__(self, gpu_id: int):
        """Initialise the mock GPU process."""
        self.gpu_id = gpu_id
        self.memory_usage = 0.0

    def get_memory_usage(self):
        """Getter for memory."""
        return self.memory_usage

    def set_memory_usage(self, usage: float):
        """Setter for memory."""
        self.memory_usage = usage


@pytest.fixture
def mock_gpu_env(monkeypatch):
    """Mock GPU environment for testing."""
    gpu_processes = {i: MockGPUProcess(i) for i in range(4)}

    def mock_check_output(cmd, *args, **kwargs):
        """Return stats in format of nvidia-smi response."""
        if "nvidia-smi" in cmd:
            gpu_id = int(cmd[-1])
            usage = gpu_processes[gpu_id].get_memory_usage()
            total = 10000  # 10GB total memory
            used = int(total * usage / 100)
            return f"{used},{total}".encode()
        return subprocess.check_output(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)
    return gpu_processes
