"""
Kairos
----------------
A flexible job scheduler for running multiple jobs across multiple GPUs.
"""


import os
import subprocess
import time
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass
from typing import Optional
from queue import Queue
import threading
import datetime
import logging
from logging.handlers import RotatingFileHandler
import json

@dataclass
class GPUSlot:
    gpu_id: int
    slot_id: int

@dataclass
class Job:
    """Represents a job to be run."""
    command: str
    env: Optional[dict] = None
    venv_path: Optional[str] = None
    working_dir: Optional[str] = None
    job_name: Optional[str] = None

class JobLogger:
    """Handles structured logging for the GPU scheduler."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.setup_main_logger()

    def setup_main_logger(self):
        """Setup the main scheduler logger."""
        self.logger = logging.getLogger('kairos')
        self.logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (rotating log file)
        main_log_file = self.log_dir / 'kairos.log'
        file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_job_logger(self, job: Job, job_index: int) -> tuple[logging.Logger, Path]:
        """Create a dedicated logger for a specific job."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = job.job_name or f"job_{job_index}"
        log_file = self.log_dir / f"{job_name}_{timestamp}.log"

        # Create job-specific logger
        logger = logging.getLogger(f'kairos.job.{job_name}.{job_index}')
        logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger, log_file

class GPUScheduler:
    def __init__(
        self,
        n_gpus: int,
        jobs_per_gpu: int,
        log_dir: str = "logs",
        memory_threshold: float = 49.0,
        check_interval: float = 5.0
    ):
        self.n_gpus = n_gpus
        self.jobs_per_gpu = jobs_per_gpu
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.log_dir = Path(log_dir)

        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logger
        self.job_logger = JobLogger(self.log_dir)
        self.logger = self.job_logger.logger

        # Create GPU slots
        self.gpu_slots = [
            GPUSlot(gpu_id=gpu_id, slot_id=slot_id)
            for gpu_id in range(n_gpus)
            for slot_id in range(jobs_per_gpu)
        ]

        self.total_jobs = 0
        self.completed_jobs = 0

        self.logger.info(f"Initialized GPU Scheduler with {n_gpus} GPUs and {jobs_per_gpu} jobs per GPU")
        self.logger.info(f"Log directory: {self.log_dir}")

    def get_gpu_memory_usage(self, gpu_id: int) -> float:
        """Get memory usage percentage for specified GPU."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,nounits,noheader",
                "-i", str(gpu_id)
            ]
            output = subprocess.check_output(cmd).decode('utf-8').strip()
            used, total = map(float, output.split(','))
            usage = (used / total) * 100
            self.logger.debug(f"GPU {gpu_id} memory usage: {usage:.1f}%")
            return usage
        except Exception as e:
            self.logger.error(f"Error getting GPU {gpu_id} memory usage: {e}")
            return 100.0

    def wait_for_gpu_memory(self, gpu_id: int, logger: logging.Logger):
        """Wait until GPU memory usage is below threshold."""
        while True:
            usage = self.get_gpu_memory_usage(gpu_id)
            if usage < self.memory_threshold:
                return
            logger.info(f"GPU {gpu_id} memory usage at {usage:.1f}% > {self.memory_threshold}%, waiting...")
            time.sleep(self.check_interval)

    def run_job(self, gpu_slot: GPUSlot, job: Job, job_index: int) -> int:
        """Run a single job on specified GPU slot and log output to file."""
        # Get job-specific logger
        job_logger, log_file = self.job_logger.get_job_logger(job, job_index)

        # Log job start
        job_info = {
            "job_index": job_index,
            "job_name": job.job_name or f"job_{job_index}",
            "gpu_id": gpu_slot.gpu_id,
            "slot_id": gpu_slot.slot_id,
            "command": job.command,
            "working_dir": str(job.working_dir or os.getcwd()),
            "start_time": datetime.datetime.now().isoformat()
        }

        job_logger.info(f"Starting job with configuration: {json.dumps(job_info, indent=2)}")

        self.wait_for_gpu_memory(gpu_slot.gpu_id, job_logger)

        # Prepare environment
        env = os.environ.copy()
        if job.env:
            env.update(job.env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot.gpu_id)

        # Construct command
        cmd = job.command
        if job.venv_path:
            activate_path = Path(job.venv_path) / "bin" / "activate"
            cmd = f"source {activate_path} && {cmd}"

        job_logger.info(f"Executing command: {cmd}")

        # Run the job and capture output
        process = subprocess.Popen(
            cmd,
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            executable='/bin/bash',
            cwd=job.working_dir,
            bufsize=1
        )

        # Stream output to log
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                job_logger.info(line.rstrip())

        return_code = process.wait()

        # Log job completion
        completion_info = {
            "end_time": datetime.datetime.now().isoformat(),
            "return_code": return_code,
            "status": "Success" if return_code == 0 else "Failed"
        }
        job_logger.info(f"Job completed: {json.dumps(completion_info, indent=2)}")

        return return_code

    def run_jobs(self, jobs: list[Job]) -> list[int]:
        """Run multiple jobs across available GPU slots."""
        total_slots = len(self.gpu_slots)
        self.logger.info(f"Starting {len(jobs)} jobs across {self.n_gpus} GPUs ({self.jobs_per_gpu} jobs per GPU)")

        # Create queues for jobs and available slots
        job_queue = Queue()
        slot_queue = Queue()

        self.total_jobs = len(jobs)

        # Put all slots in the queue
        for slot in self.gpu_slots:
            slot_queue.put(slot)

        # Track results and active jobs
        results = [None] * len(jobs)
        active_jobs = set()
        job_lock = threading.Lock()

        def worker():
            while True:
                try:
                    slot = slot_queue.get(timeout=1)
                except Queue.Empty:
                    continue

                try:
                    job_index, job = job_queue.get_nowait()
                except Queue.Empty:
                    slot_queue.put(slot)
                    break

                try:
                    with job_lock:
                        active_jobs.add(job_index)
                    result = self.run_job(slot, job, job_index)
                    results[job_index] = result
                    self.completed_jobs += 1
                    self.logger.info(f"Job {job_index} completed with status: {'Success' if result == 0 else 'Failed'}")
                finally:
                    with job_lock:
                        active_jobs.remove(job_index)
                    slot_queue.put(slot)

        # Create thread pool and run jobs
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_slots) as executor:
            futures = []

            # Queue all jobs with their indices
            for i, job in enumerate(jobs):
                job_queue.put((i, job))

            # Start workers
            for _ in range(total_slots):
                futures.append(executor.submit(worker))

            # Wait for all workers to complete
            concurrent.futures.wait(futures)

            self.logger.info("\nAll jobs completed!")
            self.logger.info("Results summary:")
            for i, result in enumerate(results):
                status = "Success" if result == 0 else f"Failed (code {result})"
                self.logger.info(f"Job {i}: {status}")

            return results

def main():
    # Example usage
    scheduler = GPUScheduler(
        n_gpus=4,
        jobs_per_gpu=2,
        log_dir="gpu_job_logs"
    )

    # Example jobs
    jobs = [
        Job(
            command="python train.py --model model1",
            venv_path=".venv",
            working_dir="/path/to/project",
            job_name="train_model1"
        ),
        Job(
            command="python train.py --model model2",
            env={"SOME_ENV_VAR": "2"},
            job_name="train_model2"
        ),
    ]

    results = scheduler.run_jobs(jobs)

if __name__ == "__main__":
    main()