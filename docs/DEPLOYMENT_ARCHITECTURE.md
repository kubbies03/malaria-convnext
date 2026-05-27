# Deployment Architecture

## Objective

Convert an existing CNN workflow into a deployment-oriented package for Jetson Nano that is easy to validate, operate, and explain in an interview.

The current repository layout separates primary deployment assets from training reference material and older legacy artifacts.

## End-to-End Flow

```text
Training / Export Environment
    |
    | train or prepare model weights
    v
Model Packaging
    |
    | optimize_for_jetson.py
    | export TorchScript artifact
    v
Jetson Nano Setup
    |
    | install dependencies
    | verify CUDA and PyTorch
    v
Runtime Execution
    |
    | single-image inference
    | folder inference
    | repeated benchmark runs
    v
Deployment Validation
```

## Main Components

### Model preparation

- source weights from the training workflow
- optional model optimization step for edge deployment
- deployment artifact intended to be loaded directly on Jetson

### Board environment

- Jetson Nano 4GB
- JetPack 4.6+
- Ubuntu 18.04
- Python runtime with PyTorch and common image-processing dependencies

### Runtime validation

- confirm Python and OS information
- confirm disk and memory availability
- confirm PyTorch import
- confirm CUDA availability before running inference

### Inference layer

- command-line execution
- single image mode
- folder mode
- CPU fallback when needed

### Benchmark layer

- warmup before timed runs
- repeated latency sampling
- summary metrics such as mean, median, min, max, and FPS
- GPU memory reporting when CUDA is active

## Engineering Decisions

- CLI entrypoints are used so the workflow is scriptable and easier to hand off.
- A system check step is included because deployment failures often come from environment mismatch, not model code.
- The repo separates `training/`, `deploy/`, and `legacy/` materials to make the review path clearer for hiring.

## Why This Architecture Matters

This project is useful for a System Engineer profile because it shows ownership across:

- deployment setup
- runtime validation
- performance verification
- operational documentation
- packaging research output into an executable workflow
