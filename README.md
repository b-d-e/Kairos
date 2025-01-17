# [kairos](https://en.wikipedia.org/wiki/Kairos): GPU Job Scheduler

[![precommit](https://github.com/b-d-e/kairos/actions/workflows/precommit.yml/badge.svg)](https://github.com/b-d-e/Kairos/actions/workflows/precommit.yml)
[![pytest](https://github.com/b-d-e/kairos/actions/workflows/tests.yml/badge.svg)](https://github.com/b-d-e/Kairos/actions/workflows/tests.yml)

A Python package for queueing and running many experiments across a local GPU cluster.

Think of it as a local, single-user equivalent to SLURM.

The scheduler monitors GPU memory usage and automatically manages job distribution to make efficient use of GPU resources.

![Kairos Animation](kairos.svg)

## Roadmap

I'd like to implemenet a more rigorous version of this in Rust at some point in future, possibly with a persistent queue on disk.

It would also be nice to run as a system service - currently have to use `tmux` or equivilant to leave running in background.

## Features

- 🎮 Support for multiple GPUs with configurable jobs per GPU
- 📊 GPU memory limits with configurable thresholds
- 🔄 Automatic job queuing and distribution
- 🐍 Virtual environment support
- 🌍 Custom envfironment variables per job
- 📝 Structured logging with rotation and configurable levels
- 🧪 Comprehensive test suite with GPU and CPU (mock) tests
- 🚀 Easy to integrate into existing projects

## Installation

1. Clone the repository:
```bash
git clone https://github.com/b-d-e/kairos.git
cd kairos
```

2. Verify you have CUDA drivers installed and accessible (tested on CUDA v12):
```bash
nvidia-smi
```

3. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows
```

4. Install the package:
```bash
# For basic installation
pip install .

# For development (includes testing tools)
pip install -e ".[dev]"
```

## Quick Start

```python
from kairos import GPUScheduler, Job

# Initialise scheduler with 4 GPUs, 2 jobs per GPU
scheduler = GPUScheduler(n_gpus=4, jobs_per_gpu=2)

# Define your jobs - in pracitce, probably dynamically
jobs = [
    Job(
        command="python train.py --config config1.yaml",
        venv_path=".venv",
        working_dir="/path/to/project",
        job_name="train_model1"  # Optional name for better log identification
    ),
    Job(
        command="python train.py --config config2.yaml",
        env={"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:5000"},
        job_name="train_model2"
    )
]

# Run all jobs
results = scheduler.run_jobs(jobs)
```

## Package Structure

```
kairos/
├── src/
│   └── kairos/
│       ├── __init__.py
│       ├── models.py        # Data models (Job, GPUSlot)
│       ├── logging.py       # Logging functionality
│       ├── scheduler.py     # Main scheduler implementation
│       └── gpu.py       # GPU utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Test fixtures
│   ├── test_scheduler_cpu.py
│   └── test_scheduler_gpu.py
└── pyproject.toml          # Package configuration
```

## Development

### Setting up the development environment

```bash
# Clone the repository
git clone https://github.com/b-d-e/kairos.git
cd kairos

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run CPU-only tests
pytest -v -m "cpu"

# Run GPU tests (requires CUDA)
pytest -v -m "gpu"

# Run all tests with coverage
pytest -v --cov=kairos
```

## Detailed Usage

### Job Configuration

Each job can be configured with:
```python
Job(
    command="your_command",           # Required: Command to run
    env={"KEY": "VALUE"},            # Optional: Additional environment variables
    venv_path="/path/to/venv",       # Optional: Virtual environment path
    working_dir="/path/to/workdir",  # Optional: Working directory for the job
    job_name="my_job"                # Optional: Name for log identification
)
```

### Scheduler Configuration

```python
scheduler = GPUScheduler(
    n_gpus=4,                # Number of GPUs available
    jobs_per_gpu=2,          # Jobs per GPU
    check_interval=5.0,      # Seconds between memory checks
    log_dir="logs"           # Directory for log files
)
```

## Logging

The scheduler now uses a structured logging system with:
- Rotating log files with size limits
- Separate log files for each job
- JSON-formatted metadata
- Configurable log levels

Log files are organised as:
- `logs/kairos.log` - Main scheduler log (with rotation)
- `logs/<job_name>_<timestamp>.log` - Individual job logs

## Memory Management

The scheduler automatically manages GPU memory allocation to ensure efficient resource utilization:

- Automatically calculates memory thresholds based on GPU capacity and jobs_per_gpu
- Reserves 95% of total GPU memory for jobs (5% buffer for system overhead)
- Evenly divides available memory between concurrent jobs
- Waits for sufficient memory to be available before starting new jobs
- Configurable check interval (default: 5 seconds)
- Proper error handling for GPU queries
- Detailed memory usage logging

For example, on a 24GB GPU with jobs_per_gpu=2:
- Total usable memory: 22.8GB (95% of 24GB)
- Memory per job: 11.4GB
- New jobs wait until at least 11.4GB is available

## Requirements

- Python 3.9+
- NVIDIA GPUs with CUDA drivers
- `nvidia-smi` available in PATH

## License

This project is licensed under the MIT License - see the LICENSE file for details.
