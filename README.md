# ProteinMap

ProteinMap is a Python toolkit for training and comparing models that predict where proteins are located within cells. It provides data loaders, model adapters, and evaluation metrics for cell images and protein-level results.

The repository does not include datasets or trained model weights. You can run the tests without them. Training requires compatible data, and evaluation also requires a checkpoint.

## Get started

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The tests use synthetic data and do not download datasets or checkpoints. To run the training or evaluation scripts, install the additional dependencies in `requirements-research.txt`:

```bash
python -m pip install -r requirements-research.txt
python scripts/train.py --help
python scripts/evaluate.py --help
```

The scripts accept paths to your data, split files, sequence cache, and model checkpoint through command-line options. Model configurations are in [`configs/models/`](configs/models/). Keep local inputs and generated outputs in directories such as `data/`, `checkpoints/`, and `results/`, which Git ignores.

## Find your way around

- [`proteinmap/data/`](proteinmap/data/) loads images, sequences, labels, and dataset splits.
- [`proteinmap/models/`](proteinmap/models/) defines the model interface, model components, and adapters for other predictors.
- [`proteinmap/eval/`](proteinmap/eval/) runs evaluations and builds result tables.
- [`proteinmap/metrics/`](proteinmap/metrics/) computes classification, image reconstruction, and protein-level summary metrics.
- [`scripts/train.py`](scripts/train.py) and [`scripts/evaluate.py`](scripts/evaluate.py) are the main command-line entry points.

The optional PUPS and ProtGPS adapters need their respective external model code and weights.

## License

ProteinMap is distributed under the [GNU General Public License, version 3 or later](LICENSE). Some data-loading and adapter code is adapted from [PUPS](https://github.com/uhlerlab/PUPS); those files identify their source in comments.
