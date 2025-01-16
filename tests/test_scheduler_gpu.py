"""Tests for GPU scheduler."""

import pytest

from kairos import GPUScheduler, Job


@pytest.mark.gpu
def test_real_gpu_execution(tmp_path):
    """Test running a real job on a real GPU."""
    scheduler = GPUScheduler(n_gpus=1, jobs_per_gpu=1, log_dir=str(tmp_path))

    # Simple CUDA test job
    test_script = """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    """

    with open(tmp_path / "cuda_test.py", "w") as f:
        f.write(test_script)

    jobs = [
        Job(command=f"python {tmp_path}/cuda_test.py", job_name="cuda_test")
    ]

    results = scheduler.run_jobs(jobs)
    assert all(result == 0 for result in results)


@pytest.mark.gpu
def test_gpu_memory_usage(tmp_path):
    """Test getting GPU memory usage via nvidia-smi."""
    scheduler = GPUScheduler(
        n_gpus=1,
        jobs_per_gpu=1,
        log_dir=str(tmp_path),
        memory_threshold=95.0,  # Set high threshold
    )

    usage = scheduler.get_gpu_memory_usage(0)
    assert isinstance(usage, float)
    assert 0 <= usage <= 100
