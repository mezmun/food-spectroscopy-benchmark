# Food Spectroscopy Regression Benchmark

Reproducible benchmarking pipeline for food near-infrared (NIR) spectroscopy regression. The benchmark compares PLSR, Ridge, SVR, ANN, and 1D-CNN models across nine regression tasks drawn from five public spectroscopy datasets.

## Evaluation design

The pipeline produces two complementary validation tracks.

- **Nested cross-validation:** 3-fold outer cross-validation with 3-fold inner model selection. Preprocessing is fitted only on the corresponding training fold.
- **Locked independent test:** a fixed hold-out subset is removed before model selection. Hyperparameters are selected only from the development subset, after which the selected model is refit and evaluated on the locked test samples.

For ANN and 1D-CNN models, target scaling is fitted only on training data. Repeated final fits quantify stochastic variability. An internal validation subset is used to estimate training duration; the selected architecture is then rebuilt and refit on the complete available training set before external prediction.

The script also generates paired prediction-error randomization tests, independent-test and development-set learning curves, and ANN/1D-CNN complexity summaries.

## Models and search spaces

The default candidate grids match the benchmark design:

- PLSR: 5, 10, 15, 20, 25, 30 latent components
- Ridge: alpha = 1e-4, 1e-3, 1e-2, 1e-1, 1, 10
- SVR (RBF): C = 0.3, 1, 3, 10, 30, 100; gamma = `scale`; epsilon = 0.1
- ANN: one- and two-hidden-layer candidates
- 1D-CNN: compact filter/kernel combinations with global average pooling

## Data

The source datasets are not redistributed here. Place the five required files in `data/` using the filenames described in `data/README.md`.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run the complete benchmark

```bash
python benchmark.py
```

Useful options:

```bash
# Fully self-contained synthetic end-to-end run
python benchmark.py --synthetic --smoke-test

# Synthetic check without TensorFlow models
python benchmark.py --synthetic --smoke-test --disable-dl

# Change the fixed test fraction and seed
python benchmark.py --holdout-fraction 0.20 --holdout-seed 2026

# Skip learning curves
python benchmark.py --skip-learning-curves
```

## Generated outputs

A complete run writes results under `outputs/revision_results/`:

- `raw/`: complete-precision CSV files, predictions, split manifests, candidate scores, and statistical tests
- `tables/revision_tables.xlsx`: consolidated spreadsheet of manuscript-relevant results
- `latex/latex_tables.txt`: LaTeX code for quantitative tables
- `figures/Figure1_complexity_panels.{png,pdf}`: ANN/1D-CNN validation error versus approximate computational cost
- `figures/Figure2_independent_test_learning_curves.{png,pdf}`: learning curves evaluated on the same locked test set
- `figures/FigureS1_cv_learning_curves.{png,pdf}`: supplementary development-set train/CV learning curves
- `run_config.json`: full run settings and dataset slice definitions

Raw numerical files retain full precision. LaTeX tables use uncertainty-aware rounding for presentation.

## Reproducibility safeguards

Before training, each task is checked for invalid slices, non-finite values, constant targets, exact target duplication among predictors, and duplicate spectra. Dataset dimensions and the development/test split are exported rather than entered manually into result tables.
