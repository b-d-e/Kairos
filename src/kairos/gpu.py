"""GPU utility functions, interfacing with nvidia-smi."""

import logging
import subprocess
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def get_gpu_memory_usage(
    gpu_id: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Get memory usage for specified GPU.

    Args:
        gpu_id: The ID of the GPU to check

    Returns:
        Tuple of (used memory in MB, total memory in MB)
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,nounits,noheader",
            "-i",
            str(gpu_id),
        ]
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        used, total = map(float, output.split(","))
        return used, total
    except Exception as e:
        logger.error(f"Error getting GPU {gpu_id} memory usage: {e}")
        return None, None


def calculate_memory_percentage(used: float, total: float) -> float:
    """Calculate memory usage percentage.

    Args:
        used: Used memory in MB
        total: Total memory in MB

    Returns:
        Memory usage percentage
    """
    if not total:
        return 100.0
    return (used / total) * 100


def calculate_per_job_memory(total_memory: float, jobs_per_gpu: int) -> float:
    """Calculate memory threshold per job based on total GPU memory.

    Reserves 95% of total memory divided by number of jobs per GPU.

    Args:
        total_memory: Total GPU memory in MB
        jobs_per_gpu: Number of concurrent jobs per GPU

    Returns:
        Memory threshold in MB per job
    """
    return (total_memory * 0.95) / jobs_per_gpu


def wait_for_gpu_memory(
    gpu_id: int,
    jobs_per_gpu: int,
    check_interval: float,
    logger: logging.Logger,
):
    """Wait until GPU has enough free memory for the job.

    Args:
        gpu_id: The GPU ID to monitor
        jobs_per_gpu: Number of concurrent jobs per GPU
        check_interval: Time to wait between checks in seconds
        logger: Logger instance to use for status updates
    """
    while True:
        used, total = get_gpu_memory_usage(gpu_id)
        if used is None or total is None:
            logger.error(f"Could not get memory usage for GPU {gpu_id}")
            return

        per_job_threshold = calculate_per_job_memory(total, jobs_per_gpu)
        available_memory = total - used

        if available_memory >= per_job_threshold:
            return

        logger.info(
            f"GPU {gpu_id} has {available_memory:.1f}MB free, "
            f"needs {per_job_threshold:.1f}MB, waiting..."
        )
        time.sleep(check_interval)
