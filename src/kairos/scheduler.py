"""Main module for GPU job scheduling and execution."""

import concurrent.futures
import datetime
import json
import os
import subprocess
import threading
from pathlib import Path
from queue import Queue
from typing import List

from .gpu import wait_for_gpu_memory
from .logging import JobLogger
from .models import GPUSlot, Job


class GPUScheduler:
    """Manages execution of multiple jobs across multiple GPUs."""

    def __init__(
        self,
        n_gpus: int,
        jobs_per_gpu: int,
        log_dir: str = "logs",
        memory_threshold: float = 49.0,
        check_interval: float = 5.0,
    ):
        """Initialise the GPU scheduler with specified configuration."""
        self.n_gpus = n_gpus
        self.jobs_per_gpu = jobs_per_gpu
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.job_logger = JobLogger(self.log_dir)
        self.logger = self.job_logger.logger

        self.gpu_slots = [
            GPUSlot(gpu_id=gpu_id, slot_id=slot_id)
            for gpu_id in range(n_gpus)
            for slot_id in range(jobs_per_gpu)
        ]

        self.total_jobs = 0
        self.completed_jobs = 0

        self.logger.info(
            (
                f"Initialized GPU Scheduler with {n_gpus}",
                f"GPUs and {jobs_per_gpu} jobs per GPU",
            )
        )
        self.logger.info(f"Log directory: {self.log_dir}")

    def run_job(self, gpu_slot: GPUSlot, job: Job, job_index: int) -> int:
        """Run a single job on specified GPU slot and log output."""
        job_logger, log_file = self.job_logger.get_job_logger(job, job_index)

        job_info = {
            "job_index": job_index,
            "job_name": job.job_name or f"job_{job_index}",
            "gpu_id": gpu_slot.gpu_id,
            "slot_id": gpu_slot.slot_id,
            "command": job.command,
            "working_dir": str(job.working_dir or os.getcwd()),
            "start_time": datetime.datetime.now().isoformat(),
        }

        job_logger.info(
            f"Starting job with configuration: \
                {json.dumps(job_info, indent=2)}"
        )

        wait_for_gpu_memory(
            gpu_slot.gpu_id,
            self.memory_threshold,
            self.check_interval,
            job_logger,
        )

        self.job_logger.log_job_start(
            job_logger, job, job_index, gpu_slot.gpu_id, gpu_slot.slot_id
        )

        env = os.environ.copy()
        if job.env:
            env.update(job.env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_slot.gpu_id)

        cmd = job.command
        if job.venv_path:
            activate_path = Path(job.venv_path) / "bin" / "activate"
            cmd = f"source {activate_path} && {cmd}"

        job_logger.info(f"Executing command: {cmd}")

        process = subprocess.Popen(
            cmd,
            env=env,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            executable="/bin/bash",
            cwd=job.working_dir,
            bufsize=1,
        )

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                job_logger.info(line.rstrip())

        return_code = process.wait()

        completion_info = {
            "end_time": datetime.datetime.now().isoformat(),
            "return_code": return_code,
            "status": "Success" if return_code == 0 else "Failed",
        }
        job_logger.info(
            f"Job completed: {json.dumps(completion_info, indent=2)}"
        )
        self.job_logger.log_job_completion(
            job_logger, job, job_index, gpu_slot.gpu_id, return_code
        )

        return return_code

    def run_jobs(self, jobs: List[Job]) -> List[int]:
        """Run multiple jobs across available GPU slots."""
        total_slots = len(self.gpu_slots)
        self.logger.info(
            f"Starting {len(jobs)} jobs across {self.n_gpus} "
            f"GPUs ({self.jobs_per_gpu} jobs per GPU)"
        )

        job_queue = Queue()
        slot_queue = Queue()
        self.total_jobs = len(jobs)
        self.job_logger.total_jobs = self.total_jobs

        for slot in self.gpu_slots:
            slot_queue.put(slot)

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
                    self.logger.info(
                        f"Job {job_index} completed with status: "
                        f"{'Success' if result == 0 else 'Failed'}"
                    )
                finally:
                    with job_lock:
                        active_jobs.remove(job_index)
                    slot_queue.put(slot)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=total_slots
        ) as executor:
            futures = []

            for i, job in enumerate(jobs):
                job_queue.put((i, job))

            for _ in range(total_slots):
                futures.append(executor.submit(worker))

            concurrent.futures.wait(futures)

            self.logger.info("\nAll jobs completed!")
