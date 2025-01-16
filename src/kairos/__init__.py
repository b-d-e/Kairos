"""Init file for kairos package."""

from .models import GPUSlot, Job
from .scheduler import GPUScheduler

__version__ = "0.1.0"
__all__ = ["GPUScheduler", "Job", "GPUSlot"]
