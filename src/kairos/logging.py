"""Logging utilties, both for the scheduler and individual jobs."""

import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .models import Job


class JobLogger:
    """Handles structured logging for the GPU scheduler."""

    def __init__(self, log_dir: Path):
        """Initialise log directory."""
        self.log_dir = log_dir
        self.setup_main_logger()
        self.total_jobs = 0

    def setup_main_logger(self):
        """Set up the main scheduler logger for scheduling events only."""
        self.logger = logging.getLogger("kairos.scheduler")
        self.logger.setLevel(logging.INFO)

        # Create formatter for scheduler events
        scheduler_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler (rotating log file)
        main_log_file = self.log_dir / "kairos.log"
        file_handler = RotatingFileHandler(
            main_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(scheduler_formatter)
        self.logger.addHandler(file_handler)

        # Console handler for scheduler events
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(scheduler_formatter)
        self.logger.addHandler(console_handler)

    def get_job_logger(
        self, job: Job, job_index: int
    ) -> tuple[logging.Logger, Path]:
        """Create a dedicated logger for a specific job."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = job.job_name or f"job_{job_index}"
        log_file = self.log_dir / f"{job_name}_{timestamp}.log"

        # Create job-specific logger
        logger = logging.getLogger(f"kairos.job.{job_name}.{job_index}")
        logger.setLevel(logging.INFO)
        logger.propagate = False  # Prevent propagation to parent loggers

        # Job-specific formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler for job output
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger, log_file

    def log_job_start(
        self,
        logger: logging.Logger,
        job: Job,
        job_index: int,
        gpu_id: int,
        slot_id: int,
    ):
        """Log job start information to both scheduler and job logs."""
        job_info = {
            "event": "job_start",
            "job_index": job_index,
            "job_name": job.job_name or f"job_{job_index}",
            "gpu_id": gpu_id,
            "slot_id": slot_id,
            "command": job.command,
            "working_dir": str(job.working_dir),
            "start_time": datetime.datetime.now().isoformat(),
        }

        # Log to scheduler log (minimal info)
        self.logger.info(
            f"Starting job {job_index}/{self.total_jobs} "
            f"({job_info['job_name']}) on GPU {gpu_id} slot {slot_id}"
        )

        # Log full details to job log
        logger.info(f"Job configuration: {json.dumps(job_info, indent=2)}")

    def log_job_completion(
        self,
        logger: logging.Logger,
        job: Job,
        job_index: int,
        gpu_id: int,
        return_code: int,
    ):
        """Log job completion information to both scheduler and job logs."""
        completion_info = {
            "event": "job_completion",
            "job_index": job_index,
            "job_name": job.job_name or f"job_{job_index}",
            "gpu_id": gpu_id,
            "end_time": datetime.datetime.now().isoformat(),
            "return_code": return_code,
            "status": "Success" if return_code == 0 else "Failed",
        }

        # Log to scheduler log (minimal info)
        self.logger.info(
            f"Job {job_index}/{self.total_jobs} "
            f"({completion_info['job_name']}) completed "
            f"with status: {completion_info['status']}"
        )

        # Log full details to job log
        logger.info(f"Job completed: {json.dumps(completion_info, indent=2)}")

    def log_gpu_wait(self, gpu_id: int, usage: float, threshold: float):
        """Log GPU waiting status to scheduler log only."""
        self.logger.info(
            f"Waiting for GPU {gpu_id} memory usage to drop below \
                {threshold}% (currently {usage:.1f}%)"
        )

    def log_scheduler_event(self, message: str, level: str = "info"):
        """Log scheduler-specific events."""
        log_func = getattr(self.logger, level.lower())
        log_func(message)
