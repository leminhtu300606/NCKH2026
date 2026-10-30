# Repository initialization

This repository contains the source code for the UAV Privacy-Preserving Video Surveillance system.

## Directory layout
```
UAV-Privacy/
├─ configs/               # YAML configuration files
├─ data/
│   ├─ raw/               # read‑only raw dataset (Kaggle mounts)
│   ├─ interim/           # temporary conversion files (COCO, splits)
│   └─ processed/         # final manifests, stats, features
├─ src/                   # source modules
│   ├─ data/               # dataset loaders
│   ├─ detection/          # YOLO wrapper
│   ├─ tracking/           # ByteTrack integration
│   ├─ features/           # 21‑dim extractor
│   ├─ labeling/           # Re‑ID attack pipeline
│   ├─ models/             # model checkpoints & export
│   ├─ policy/             # policy dispatcher & hysteresis
│   ├─ anonymization/      # maskers
│   └─ safety/             # fail‑safe gate
├─ scripts/               # CLI utilities (prepare, train, export, ...)
├─ tests/                 # unit tests
├─ notebooks/             # exploratory notebooks
├─ artifacts/             # generated models, ONNX, visual samples
└─ reports/               # markdown evaluation reports
```

## Quick start (local)
```bash
# create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
```
