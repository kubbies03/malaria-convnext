# Jetson Nano Deployment Workflow

This directory contains the deployment-facing runtime package for the project.

It is derived from the working deployment materials already present in the repository and reorganized into a cleaner structure for review and handoff.

## Goals

- provide a simple board-side execution path
- validate the target environment before inference
- keep inference and benchmark commands scriptable
- present the project in a deployment-oriented format

## Files

- `install_jetson.sh`: dependency installation helper
- `system_check.py`: target board validation script
- `infer.py`: TorchScript inference CLI
- `benchmark.py`: repeatable latency benchmark CLI

## Typical Deployment Flow

1. Copy repository or deployment folder to Jetson Nano.
2. Install dependencies.
3. Run system validation.
4. Transfer or export the model artifact.
5. Run single-image inference.
6. Run benchmark for performance reporting.

## Example Commands

```bash
python3 system_check.py
python3 infer.py --model ./models/model.pt --image ../../samples/Infected_sample.png
python3 benchmark.py --model ./models/model.pt --image ../../samples/Infected_sample.png --runs 100
```

## Why this directory exists

The original project files already showed a successful deployment workflow, but they were mixed into notebook-oriented content. This folder isolates the deployment path so reviewers can understand the system engineering side quickly.
