"""Logging utilities."""

import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .models import Job


class JobLogger:
    """Handles structured logging for the GPU scheduler."""

    def __init__(self, log_dir: Path):
        """Initialse the JobLogger with a log directory."""
        self.log_dir = log_dir
        self.setup_main_logger()

    def setup_main_logger(self):
        """Set up the main scheduler logger."""
        self.logger = logging.getLogger("kairos")
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        main_log_file = self.log_dir / "kairos.log"
        file_handler = RotatingFileHandler(
            main_log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_job_logger(
        self, job: Job, job_index: int
    ) -> tuple[logging.Logger, Path]:
        """Create a dedicated logger for a specific job."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = job.job_name or f"job_{job_index}"
        log_file = self.log_dir / f"{job_name}_{timestamp}.log"

        logger = logging.getLogger(f"kairos.job.{job_name}.{job_index}")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger, log_file
