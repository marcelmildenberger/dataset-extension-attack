# Neural Pattern Learning (NEPAL) Attack Thesis Repository

This repository accompanies the thesis on the Neural Pattern Learning (NEPAL) Attack. It links to the implementation repository and provides documentation for reproducing the experiments that evaluate the attack pipeline against privacy-preserving record linkage (PPRL) systems.

## Project Overview

The NEPAL pipeline models an adversary who extends an existing encoded dataset with auxiliary records to recover sensitive information. The attack implementation (referenced in the thesis and hosted at [`marcelmildenberger/dataset-extension-attack`](https://github.com/marcelmildenberger/dataset-extension-attack)) performs four major stages:

1. **Data ingestion and preprocessing** – prepares encoded linkage datasets and auxiliary data.
2. **Model training and hyperparameter optimisation** – uses Ray Tune and Optuna to search neural architectures that reconstruct plaintext n-grams from encodings.
3. **Inference and reconstruction** – generates candidate plaintexts for the encoded records.
4. **Analysis and reporting** – produces plots and tables that quantify attack efficacy.

The `docs/` directory in this thesis repository contains detailed guides that map these stages to the corresponding scripts in the implementation repository.

## Getting Started with Docker

The easiest way to reproduce the pipeline is to run the implementation repository inside Docker. The following commands mirror the setup that was used while preparing the thesis:

```bash
git clone https://github.com/marcelmildenberger/dataset-extension-attack.git
cd dataset-extension-attack
git submodule update --init --recursive --remote

docker build -t NEPAL .
docker run --gpus all -it -v $(pwd):/usr/app NEPAL bash
```

> **Note:** GPU access is optional but strongly recommended for the hyperparameter tuning stage. The container exposes the repository at `/usr/app`.

## Running the Default Nepal Case

The repository provides a ready-made configuration for the Nepal voter dataset. After entering the container, execute:

```bash
cd /usr/app
python main.py --config configs/nepal_config.json
```

This launches the full NEPAL pipeline with default parameters, including preprocessing, model training, and reconstruction. Results are written to the directory configured inside `nepal_config.json` (see [`docs/parameters.md`](docs/parameters.md) for a full description of the configuration schema).

## Batch Experiment Setup

To queue multiple experiments or run the hyperparameter sweeps used in the thesis, leverage `experiment_setup.py`. For example, to reproduce the TabMinHash experiments you can run:

```bash
python experiment_setup.py \
  --config configs/nepal_config.json \
  --encoding tabminhash \
  --output-root outputs/tabminhash_sweeps
```

The script materialises per-experiment configuration files under the chosen output directory and can submit them directly to Ray Tune depending on the flags you enable. See [`docs/experiment_setup.md`](docs/experiment_setup.md) for more detail on its arguments and expected folder structure.

## Analysis Notebooks

Evaluation notebooks in the `analysis/` directory generate the plots and tables referenced throughout the thesis. Launch Jupyter Lab from inside the Docker container with:

```bash
jupyter lab --ip=0.0.0.0 --no-browser --port=8888
```

Then open `analysis/attack_performance.ipynb` (or the relevant notebook) in your browser via the provided token. The notebooks expect results from `main.py` or `experiment_setup.py` to be available under the output directories described in [`docs/analysis.md`](docs/analysis.md).

## Additional Documentation

* [`docs/parameters.md`](docs/parameters.md) – Detailed configuration and parameter reference.
* [`docs/experiment_setup.md`](docs/experiment_setup.md) – Instructions for orchestrating large experiment batches.
* [`docs/analysis.md`](docs/analysis.md) – Guidance for generating the analytical figures.

For questions or clarifications regarding the implementation, please refer to the code repository or reach out to the thesis author.