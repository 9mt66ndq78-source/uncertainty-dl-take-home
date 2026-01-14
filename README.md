# Deep Bayesian Active Learning

Implementation of Bayesian Active Learning methods for the Uncertainty in Deep Learning course.

## Project Structure

### src/

This folder contains all the core classes and support functions

- config.py - Global constants such as number of epochs to train
- data_funcs.py - Data loading functions
- models.py - Model definitions for Bayesian CNN, BLR, Full BLR, and MFVI
- simclr.py - Helper functions used for simclr pretraining of CNN backbone
- acquisition.py - Acquisition functions for (BALD, Entropy, etc.)
- train_funcs.py - Functions for training and evaluating the models.
- utils.py - Utilities, mainly logging to json.

### experiments/

This contains scripts used for actually running the experiments

- 01_reproduction.py - This runs the experiments that reproduce the original paper.
- 02_regression.py - This runs experiments for both the minimal extension and the novel extension, depending on what options you choose.

### analysis/

After running experiments, you can use the jupyter notebooks here for generating visuals. _Not actual notebooks, but cells embedded in python file using #%%_

- compare_acquisition.py - used to compare average results between different acquisition methods.
- qualitative_analysis.py - used to visualise what each run acquired at each acquisition cycle, and other similar functions.

### Other directories

- data/ - stores the MNIST dataset
- results/ - stores the logs from each run, each log is used to creat visualisations. /phase1/ has results from the reproduction experiments and /phase2/ has results from all the regression based experiments.
- cahce/ - stores the SimCLR pretrained CNN so it doesn't have to be trained again every time.

## Setup

Install dependencies using `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install torch torchvision numpy scikit-learn tqdm wandb matplotlib
```

## Phase 1: Classification Reproduction

Reproduces Figures 1 & 2 from Gal et al. (2017) comparing active learning acquisition strategies.

I use "uv run" to run my experiments, but you should also be able to use "python/python3" instead, if all dependecies were installed correctly.

### Quick Test

Run a quick test (5 cycles, 5 epochs):

```bash
uv run experiments/01_reproduce.py --method random --seed 42 --quick
```

### Full Experiment

Run the complete experiment (100 cycles, 50 epochs):

```bash
# Random baseline
uv run experiments/01_reproduce.py --method random --seed 42

# BALD
uv run experiments/01_reproduce.py --method bald --seed 42

# Max Entropy
...
```

To see a full list of method options run

```bash
uv run experiments/01_reproduce.py --help
```

## Phase 2: Regression Baseline

Compares Analytic Bayesian Linear Regression (BLR) vs Mean-Field Variational Inference (MFVI) for active learning with regression on one-hot encoded MNIST labels.

### Pretraining Strategies

| Strategy          | Description                                                   |
| ----------------- | ------------------------------------------------------------- |
| `simple_pretrain` | Train CNN once on initial labeled set (default)               |
| `simple_retrain`  | Retrain CNN from scratch at each AL cycle                     |
| `simclr`          | Self-supervised SimCLR pretraining on entire train set (once) |

### Quick Test

```bash
# BLR with simple pretrain (default)
uv run experiments/02_regression.py --method blr --seed 42 --quick

# MFVI with SimCLR pretraining
uv run experiments/02_regression.py --method mfvi --pretrain simclr --seed 42 --quick
```

### Full Experiment

```bash
# BLR - Simple pretrain (train once on initial 20 samples)
uv run experiments/02_regression.py --method blr --pretrain simple_retrain --seed 1
```

### Options

- `--method`: `blr` (Bayesian Linear Regression) or `mfvi` (Mean-Field VI)
- `--pretrain`: CNN pretraining strategy (`simple_pretrain`, `simple_retrain`, `simclr`)
- `--seed`: Random seed for reproducibility
- `--quick`: Quick test mode (5 cycles, fewer epochs)

## Results

Results are saved as JSON files in `/results/{phase}/`

## Models

### BayesianCNN (Phase 1)

CNN with MC Dropout from Gal et al. (2017):

1. Conv2d(1→32, 4×4) + ReLU
2. Conv2d(32→32, 4×4) + ReLU
3. MaxPool2d(2×2)
4. Dropout(p=0.25)
5. Flatten
6. Linear(3872→128) + ReLU
7. Dropout(p=0.5)
8. Linear(128→10) + LogSoftmax

### BayesianLinearRegression (Phase 2)

Analytic multivariate Bayesian Linear Regression with Matrix Normal posterior:

- Shared precision across all K=10 outputs
- Weight matrix: `mu` is `[D, K]`, covariance `Sigma` is `[D, D]`

### MFVIRegression (Phase 2)

Mean-Field Variational Inference for multivariate regression:

- Diagonal posterior approximation with shared variance across outputs
- Optimized via ELBO maximization (Adam optimizer)
- Weight matrix: `mu` is `[D, K]`, variance `var` is `[D]`

## Running Experiments

Here are bash scripts used to run the experiments mentioned in the report. Use `analysis/compare_acquisition.py` to generate figures.

### Gal et al. reproduction

```bash
for method in random bald entropy variation_ratios mean_std; do
  for seed in 1 2 3; do
      uv run experiments/01_reproduce.py --method $method --seed $seed
  done
done
```

### Regression Baseline

```bash
for method in independent_blr mfvi; do
  for seed in 1 2 3; do
      uv run experiments/02_regression.py --method $method --pretrain simple_retrain --acquisition variance --seed $seed
  done
done
```

### Simclr + Clustering

```bash
for method in independent_blr mfvi; do
  for acquisition in k_centroids k_max; do
    for seed in 1 2 3; do
        uv run experiments/02_regression.py --method $method --pretrain simclr --acquisition $acquisition --seed $seed
    done
  done
done
```
