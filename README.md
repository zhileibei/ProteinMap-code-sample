# ProteinMap

ProteinMap is a Python framework for comparing protein subcellular localization models on a common evaluation interface. It brings data loading, model adapters, per-cell and per-protein metrics, uncertainty estimates, and image reconstruction metrics into one package.

This is a **public code sample** exported from a research project. The data, model checkpoints, experiment outputs, unpublished analysis, and machine-specific run scripts are deliberately omitted. The code shows how the benchmark is organized; running a full biological benchmark requires obtaining the relevant datasets and model weights separately.

## What the code demonstrates

- `proteinmap/models/base.py` defines a predictor interface. `proteinmap/models/localization_predictor.py` adapts models to that interface.
- `proteinmap/models/components/` composes sequence, image, and RNA encoders with fusion and prediction heads.
- `proteinmap/eval/unified_evaluator.py` runs a predictor and records metrics at both cell and protein levels.
- `proteinmap/metrics/aggregation.py` aggregates repeated cell observations by protein and uses protein-clustered bootstrap resampling for cell-level confidence intervals.
- `proteinmap/metrics/compartment_metrics.py` and `inpainting_metrics.py` compute classification and image reconstruction metrics.
- `scripts/train.py` and `scripts/evaluate.py` are the current data-dependent entry points. `configs/models/` contains model architecture examples.

The [protein-level aggregation and bootstrap implementation](proteinmap/metrics/aggregation.py) is a useful starting point for reviewing the code. Treating each cell as an independent observation would understate uncertainty when many cells share the same protein.

## Quick check without research data

Use Python 3.10 or later:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use small synthetic arrays; they do not download data or checkpoints.

## Running with your own data

Install the model and image dependencies with `python -m pip install -r requirements-research.txt`. Pass dataset, sequence-cache, split, and checkpoint paths through the command-line flags shown by `python scripts/train.py --help` and `python scripts/evaluate.py --help`. The data loader can also use `PROTEINMAP_DATA_ROOT` and `PROTEINMAP_ESM2_CACHE`. Input data and generated outputs belong under ignored local directories such as `data/`, `checkpoints/`, and `results/`.

The optional PUPS adapter expects the [PUPS repository](https://github.com/uhlerlab/PUPS) as a sibling checkout. The optional ProtGPS adapter uses `PROTEINMAP_PROTGPS_ROOT` or a local cache directory. These adapters and the full benchmarks require external model code and weights; the synthetic tests do not.

## Provenance and license

Portions of the data-loading and benchmark-adapter code were adapted from [uhlerlab/PUPS](https://github.com/uhlerlab/PUPS), whose authors release that code under GNU GPL version 3 or later. The adapted files identify the source in their module comments. This public export is distributed under the same license; see [LICENSE](LICENSE). The repository contains no original HPA images or model checkpoints.

## Scope of this export

The complete research checkout includes exploratory scripts, plots, and planning documents. They are omitted here because they are not needed to assess the software design and may contain local paths or preliminary results. This repository starts with fresh Git history so removed material is absent from earlier commits as well.
