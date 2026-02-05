# scGRID (single-cell Gene Regulation-based IDentifier)

scGRID is a graph-based single-cell annotation framework that represents cell identity using Cell Type–Specific Gene Regulatory Networks (CTSGRNs) and classifies them using Graph Attention Networks (GATs).

This README documents **installation** and **usage of the CLI**.


## System requirements

- **Operating system**: Linux  
  (Windows users must install and use **WSL**)
- **Python**: 3.9.19
- **R**: 4.4.0
- **Conda** (recommended)
- **pip**

## Data

The **Marrow-10X_P7_3** dataset is used for model training, while **Marrow-10X_P7_2** is used as an independent test dataset.
CTSGRNs, node feature tables, trained GAT models, and example output files are uploaded on Zenodo:
https://zenodo.org/records/18420010

## Installation

### 1. Clone the repository

This script trains a graph-level GAT classifier on Cell Type-Specific Gene Regulatory Networks (CTSGRNs).
```bash
git clone https://github.com/Malleshaiah-Lab/scGRID.git
cd scGRID
```

### 2. Create the Conda environment (recommended)
```bash
conda env create -f scgrid-env.yml
conda activate scgrid-env
```

Alternatively, you may install Python dependencies using:
```bash
pip install -r scgrid-env_requirements.txt
```

### 3. Install scGRID in editable mode
```bash
pip install -e .
```

### 4. Verify installation
```bash
scgrid --help
```

## Command-line usage

scGRID provides a unified CLI with multiple subcommands.
```bash
scgrid --help
```

### Training a model
```bash
scgrid train \
  --data-root <PATH_TO_DATA_ROOT> \
  --input <INPUT_DATASET> \
  --input-type <rds|raw> \
  [additional options]
```
Example:
```bash
scgrid --data-root <PATH_TO_DATA_ROOT> \
--input <PATH_TO_RDS_OBJECT> --input-type rds \
train --do-subsample TRUE --num-subsets 10 --sampling-rate 0.7 --subset-suffix _70pRandSub
```

### Classifying new data
```bash
scgrid classify \
  --data-root <PATH_TO_DATA_ROOT> \
  --input <QUERY_DATASET> \
  --models <PATH_TO_TRAINED_MODELS_FOLDER> \
  [additional options]
```
Example:
```bash
scgrid --data-root <PATH_TO_DATA_ROOT> \
      --input <PATH_TO_RDS_OBJECT> --input-type rds \
      --tissue Marrow-10X_P7_2 \
      classify --models-dir <PATH_TO_TRAINED_MODELS_FOLDER> \
      --entropy-threshold 0.9
```

## Notes

- scGRID is designed for Linux-based environments
- R is required for preprocessing steps involving Seurat-based workflows.

*The current version of scGRID (0.1.0) is implemented and tested using Python 3.9.*

