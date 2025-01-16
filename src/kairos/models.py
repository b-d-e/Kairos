"""Data classes for Jobs and Compute."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUSlot:
    """Represents a slot on a GPU that can run a job."""

    gpu_id: int
    slot_id: int


@dataclass
class Job:
    """Represents a job to be run on a GPU.

    Attributes:
        command: Shell command to execute
        env: Optional environment variables to set
        venv_path: Optional path to virtual environment to activate
        working_dir: Optional working directory for the command
        job_name: Optional name for job identification in logs
    """

    command: str
    env: Optional[dict] = None
    venv_path: Optional[str] = None
    working_dir: Optional[str] = None
    job_name: Optional[str] = None
