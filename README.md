# CNN Deployment on Jetson Nano

Deployment-focused portfolio repository for preparing, packaging, and running CNN inference on NVIDIA Jetson Nano.

This repo is organized for a System Engineer application: it emphasizes deployment workflow, runtime validation, packaging, benchmarking, and handoff-quality documentation rather than only notebook training.

## Project Focus

- target hardware: Jetson Nano 4GB
- deployment target: edge CNN inference
- runtime style: CLI-based validation, inference, and benchmarking
- portfolio angle: deployment engineering for embedded AI

## Recommended Review Path

1. Read [docs/DEPLOYMENT_ARCHITECTURE.md](C:/Users/HP/Desktop/AI_Engineer/Project/ResNet/docs/DEPLOYMENT_ARCHITECTURE.md:1)
2. Read [deploy/jetson_nano/README.md](C:/Users/HP/Desktop/AI_Engineer/Project/ResNet/deploy/jetson_nano/README.md:1)
3. Review [deploy/jetson_nano/system_check.py](C:/Users/HP/Desktop/AI_Engineer/Project/ResNet/deploy/jetson_nano/system_check.py:1)
4. Review [deploy/jetson_nano/infer.py](C:/Users/HP/Desktop/AI_Engineer/Project/ResNet/deploy/jetson_nano/infer.py:1)
5. Review [deploy/jetson_nano/benchmark.py](C:/Users/HP/Desktop/AI_Engineer/Project/ResNet/deploy/jetson_nano/benchmark.py:1)
6. Check [docs/SYSTEM_ENGINEER_NOTES.md](C:/Users/HP/Desktop/AI_Engineer/Project/ResNet/docs/SYSTEM_ENGINEER_NOTES.md:1)

## Repository Layout

```text
ResNet/
|-- artifacts/
|   `-- image-processing-deploy-model-on-jetson-nano.zip
|-- assets/
|   `-- confusion_matrix.png
|-- deploy/
|   `-- jetson_nano/
|       |-- README.md
|       |-- install_jetson.sh
|       |-- system_check.py
|       |-- infer.py
|       `-- benchmark.py
|-- docs/
|   |-- DEPLOYMENT_ARCHITECTURE.md
|   `-- SYSTEM_ENGINEER_NOTES.md
|-- legacy/
|   `-- diabetic_retinopathy_jetson/
|       |-- README.md
|       |-- JETSON_DEPLOYMENT.md
|       |-- jetson_inference.py
|       |-- optimize_for_jetson.py
|       `-- requirements_jetson.txt
|-- samples/
|   |-- Infected_sample.png
|   `-- Uninfected_sample.png
|-- training/
|   `-- ConvNeXt.ipynb
|-- requirements.txt
`-- README.md
```

## Folder Roles

- `deploy/jetson_nano/`: cleaned deployment entrypoints for runtime validation, inference, and benchmark
- `docs/`: architecture and interview-facing notes
- `training/`: training-side notebook retained for reference
- `samples/`: small example images for demo commands
- `artifacts/`: exported package or large handoff asset
- `legacy/`: older project materials preserved for traceability but not part of the primary review path

## Quick Start

```bash
python3 deploy/jetson_nano/system_check.py
python3 deploy/jetson_nano/infer.py --model /path/to/model.pt --image samples/Infected_sample.png
python3 deploy/jetson_nano/benchmark.py --model /path/to/model.pt --image samples/Infected_sample.png --runs 100
```

## Notes

- The main project story in this repository is Jetson Nano deployment, not model research.
- The `legacy/` folder contains older diabetic retinopathy deployment materials that were kept for reference because they do not fully match the current malaria-focused deployment structure.
- The current top-level layout is intended to make portfolio review faster and reduce confusion between training, deployment, and historical artifacts.
