# UPLC Method Recommendation (HTE Reaction Analysis)

A research codebase for the manuscript **“An Intelligent Data-Driven Framework for UPLC Method Recommendation in High-Throughput Reaction Analysis”** (manuscript in preparation).

This repository contains:

* **Method-aware retention time (RT) prediction** workflows for multiple UPLC methods;
* a **reaction-level scoring and ranking** pipeline for UPLC method recommendation;
* reproducible scripts and package modules for preprocessing, model training/evaluation, similarity analysis, and method recommendation.

---

## 1) Project overview (what this repo does)

### RT prediction (molecule-level, method-aware)

For each UPLC method, the project supports retention time prediction under a fixed chromatographic context (method-specific modeling).

The current repository includes:

* raw RT-related datasets under `data/raw/`;
* processed feature tables and intermediate datasets under `data/processed/`;
* train/test split tables under `data/train_test_split/`;
* reaction-level input data under `data/reaction/`;
* SMARTS-related definitions/features under `data/smarts/`;
* traditional ML code under `src/ml/` and `scripts/ml/`;
* deep learning code under `src/gnn_bert/` and `scripts/dl/`;
* similarity analysis code under `src/similarity_analysis/`;
* method recommendation code under `src/method_recommendation/` and `scripts/method-recommendation/`.

### Method recommendation (reaction-level scoring)

Given a reaction system with multiple components, the workflow predicts the RT of each component under candidate UPLC methods, then scores and ranks these methods according to downstream separation-oriented criteria.

Typical considerations include:

1. whether predicted RTs fall into a usable retention window;
2. whether components are sufficiently separated from each other;
3. whether reaction-level priorities can be reflected in scoring.

The output is a ranked list of candidate UPLC methods, together with result files and analysis outputs.

---

## 2) Repository structure

```text
.
├── data/
│   ├── processed/              # processed feature tables and intermediate datasets
│   ├── raw/                    # raw source datasets
│   ├── reaction/               # reaction-level input tables
│   ├── smarts/                 # SMARTS patterns or related feature definitions
│   └── train_test_split/       # prepared train/test split datasets
│
├── results/
│   ├── dl/                     # deep learning outputs
│   ├── method_recommendation/  # method recommendation outputs
│   ├── ml/                     # traditional ML outputs
│   └── preprocessing/          # preprocessing outputs
│
├── scripts/
│   ├── dl/                     # runnable scripts for deep learning workflows
│   ├── method-recommendation/  # runnable scripts for method recommendation
│   ├── ml/                     # runnable scripts for traditional ML workflows
│   └── preprocessing/          # runnable scripts for preprocessing
│
├── src/
│   ├── gnn_bert/               # deep learning package
│   ├── method_recommendation/  # recommendation package
│   ├── ml/                     # traditional ML package
│   ├── similarity_analysis/    # similarity analysis package
│   └── __init__.py
│
├── environment_dl.yml          # conda environment for deep learning workflows
├── environment_ml.yml          # conda environment for ML workflows
├── setup.py                    # package installation config
├── LICENSE
└── README.md
```

> Note:
> The current repository is organized around `data/`, `results/`, `scripts/`, and `src/`.
> Older descriptions based on paths such as `datas/`, top-level `ml/`, or top-level `gnn_bert/` are no longer accurate for the current version of the repository.

---

## 3) Data format

### Raw RT tables (`data/raw/`)

Raw source data are stored under `data/raw/`.

At minimum, RT prediction datasets are expected to contain fields equivalent to:

* molecular structure information (for example, SMILES);
* retention time labels;
* method identifiers when multiple chromatographic methods are involved.

### Processed feature tables (`data/processed/`)

Processed datasets are stored under `data/processed/`.

Depending on the workflow, these tables may include:

* physicochemical descriptors;
* SMARTS / functional-group related features;
* fingerprint-based molecular representations;
* cleaned labels and intermediate columns used for model training.

### Split tables (`data/train_test_split/`)

Prepared train/test split files are stored under `data/train_test_split/`.

These are used for reproducible model development and evaluation.

### Reaction-level input tables (`data/reaction/`)

Reaction-level inputs used by the recommendation workflow are stored under `data/reaction/`.

These files are typically used to aggregate multiple reaction components and evaluate candidate UPLC methods at the system level rather than the single-molecule level.

---

## 4) Installation

> Recommended: use **conda**, especially for RDKit-related workflows.

### 4.1 Create environment from the provided files

For traditional ML workflows:

```bash
conda env create -f environment_ml.yml
conda activate <your-ml-env>
```

For deep learning workflows:

```bash
conda env create -f environment_dl.yml
conda activate <your-dl-env>
```

### 4.2 Install the package in editable mode

After creating and activating the environment:

```bash
pip install -e .
```

This allows you to import modules directly from `src/` during development.

---

## 5) Quick start

### 5.1 Prepare data

Place or organize the required datasets under the corresponding directories in `data/`:

* `data/raw/`
* `data/processed/`
* `data/train_test_split/`
* `data/reaction/`
* `data/smarts/`

### 5.2 Run preprocessing workflows

Use scripts under `scripts/preprocessing/` to prepare features, intermediate tables, or other preprocessing outputs.

### 5.3 Run model training / evaluation

Use:

* `scripts/ml/` for traditional machine learning workflows;
* `scripts/dl/` for deep learning workflows.

Core implementation code is available in:

* `src/ml/`
* `src/gnn_bert/`

### 5.4 Run similarity analysis

Use the similarity analysis module in `src/similarity_analysis/` when chemical-space comparison or similarity-based evaluation is needed.

### 5.5 Run method recommendation

Use scripts under `scripts/method-recommendation/` together with the package code in `src/method_recommendation/` to generate reaction-level UPLC method recommendations.

Outputs are written to `results/method_recommendation/`.

---

## 6) Notes

* `data/` stores inputs and prepared datasets.
* `results/` stores generated outputs.
* `scripts/` contains runnable entry points grouped by workflow.
* `src/` contains the main reusable Python package code.

In short: inputs go into `data/`, implementation lives in `src/`, runnable workflow entry points live in `scripts/`, and outputs belong in `results/`.

---

## 7) Manuscript & citation

**Manuscript:** *An Intelligent Data-Driven Framework for UPLC Method Recommendation in High-Throughput Reaction Analysis* (in preparation; not yet submitted).

If you use this repository in academic work, please cite the corresponding manuscript once the formal paper information is available.
