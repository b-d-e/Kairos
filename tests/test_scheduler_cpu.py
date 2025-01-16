"""Tests for the GPUScheduler class using mocked CPU resources."""

import pytest

from kairos import GPUScheduler, Job


def test_scheduler_initialization():
    """Test the initialization of the GPUScheduler class."""
    scheduler = GPUScheduler(n_gpus=2, jobs_per_gpu=2)
    assert scheduler.n_gpus == 2
    assert scheduler.jobs_per_gpu == 2
    assert len(scheduler.gpu_slots) == 4


@pytest.mark.cpu
def test_job_execution(mock_gpu_env, tmp_path):
    """Test running multiple jobs on the CPU scheduler."""
    scheduler = GPUScheduler(n_gpus=2, jobs_per_gpu=1, log_dir=str(tmp_path))

    # Simple echo job
    jobs = [
        Job(command="echo 'test1'", job_name="test1"),
        Job(command="echo 'test2'", job_name="test2"),
    ]

    results = scheduler.run_jobs(jobs)
    assert all(result == 0 for result in results)

    # Check log files were created
    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) >= 2  # At least one log file per job


@pytest.mark.cpu
def test_failed_job(mock_gpu_env, tmp_path):
    """Test running a job that fails."""
    scheduler = GPUScheduler(n_gpus=1, jobs_per_gpu=1, log_dir=str(tmp_path))

    jobs = [Job(command="exit 1", job_name="failing_job")]

    results = scheduler.run_jobs(jobs)
    assert results[0] == 1
