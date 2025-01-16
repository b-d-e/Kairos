# [Kairos](https://en.wikipedia.org/wiki/Kairos): GPU Job Scheduler

A simple Python utility for scheduling and running multiple jobs across a local GPU cluster.

Think of it as a a local, single-user equivilant to SLURM.

The scheduler monitors GPU memory usage and automatically manages job distribution to make efficient use of GPU resources.

> N.b. This is extremely unrigorously tested and so shouldn't be trusted for anything too mission critical right now!

>In the short term I will add packaging and a test suite. Long term, I'd like to rewrite it in Rust.

## Features

- 🎮 Support for multiple GPUs with configurable jobs per GPU
- 📊 GPU memory limits with configurable thresholds
- 🔄 Automatic job queuing and distribution
- 🐍 Virtual environment support
- 🌍 Custom environment variables per job
- 📝 Detailed logging with configurable levels
- 🚀 Easy to integrate into existing projects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/b-d-e/kairos.git
cd kairos
```

2. Verify you have CUDA drivers installed and accessible (only tested on CUDA v12):
```bash
nvidia-smi
```

That's it. Should work with vanilla Python modules.

## Quick Start

Modular usage
```python
from gpu_scheduler import GPUScheduler, Job

# Initialise scheduler with 4 GPUs, 2 jobs per GPU
scheduler = GPUScheduler(n_gpus=4, jobs_per_gpu=2)

# Define your jobs
jobs = [
    Job(
        command="python train.py --config config1.yaml",
        venv_path=".venv",
        working_dir="/path/to/project"
    ),
    Job(
        command="python train.py --config config2.yaml",
        env={"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:5000"}
    )
]

# Run all jobs
results = scheduler.run_jobs(jobs)
```

## Command Line Interface

The scheduler can be run directly from the command line:

```bash
python gpu_scheduler.py --gpus 4 --jobs-per-gpu 2 --memory-threshold 49 --check-interval 5
```

You'll need to hardcode your experiment commands.

Available arguments:
- `--gpus`: Number of GPUs to use (default: 1)
- `--jobs-per-gpu`: Number of concurrent jobs per GPU (default: 1)
- `--memory-threshold`: GPU memory threshold percentage (default: 50.0)
- `--check-interval`: Seconds between memory checks (default: 5.0)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)

## Detailed Usage

### Job Configuration

Each job can be configured with:
```python
Job(
    command="your_command",           # Required: Command to run
    env={"KEY": "VALUE"},            # Optional: Additional environment variables
    venv_path="/path/to/venv",       # Optional: Virtual environment path
    working_dir="/path/to/workdir"   # Optional: Working directory for the job
)
```

### Scheduler Configuration

```python
scheduler = GPUScheduler(
    n_gpus=4,                # Number of GPUs available
    jobs_per_gpu=2,          # Jobs per GPU
    memory_threshold=50.0,   # Wait until GPU memory is below this percentage
    check_interval=5.0,      # Seconds between memory checks
    log_level="INFO"         # Logging level
)
```

### Running Jobs with Virtual Environment

```python
jobs = [
    Job(
        command="python train.py --model resnet50",
        venv_path=".venv",
        working_dir="/path/to/project"
    )
]
```

### Custom Environment Variables

```python
jobs = [
    Job(
        command="python train.py",
        env={
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "MY_CUSTOM_VAR": "value"
        }
    )
]
```

## Memory Management

The scheduler monitors GPU memory usage and only starts new jobs when:
1. A GPU slot is available
2. The GPU's memory usage is below the specified threshold

For example, with default settings:
- Memory threshold: 50%
- The scheduler will wait until GPU memory usage drops below 50% before starting a new job
- Memory is checked every 5 seconds (configurable)

## Logging

The scheduler provides detailed logging about job execution:
```python
scheduler = GPUScheduler(log_level="DEBUG")  # For most detailed logging
```

Log levels:
- `DEBUG`: Detailed information about memory usage and job execution
- `INFO`: General progress and status updates
- `WARNING`: Warning messages
- `ERROR`: Error messages only

## Error Handling

The scheduler provides robust error handling:
- Captures and logs process output
- Reports GPU memory monitoring errors
- Handles job failures gracefully
- Returns process exit codes for all jobs

## Requirements

- Python 3.9+
- NVIDIA GPUs
- `nvidia-smi` / CUDA drivers


## License

This project is licensed under the MIT License - see the LICENSE file for details.
