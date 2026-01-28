# scGRID (single-cell Gene Regulation-based IDentifier)

scGRID is a graph-based single-cell annotation framework that represents cell identity using Cell Type–Specific Gene Regulatory Networks (CTSGRNs) and classifies them using Graph Attention Networks (GATs).

## Repository contents

This version includes three standalone Python scripts:

1. scgrid_train.py – training of GAT classifiers on CTSGRNs
2. scgrid_test.py – classification of query CTSGRNs using trained GAT models
3. scgrid_majority_vote.py – ensemble majority voting across k trained models

No command-line interface or end-to-end pipeline is provided yet.

## 1. Training a GAT classifier

### (scgrid_train.py)

This script trains a graph-level GAT classifier on Cell Type-Specific Gene Regulatory Networks (CTSGRNs).

### Input

- A list of CTSGRN folder paths, each corresponding to a random subset used for k-fold cross-validation
- A list of node feature folder paths, containing the following node features:
  1. average gene expression
  2. fold change
  3. pct.1
  4. pct.2

Each CTSGRN is converted into a PyTorch Geometric graph, where:
- nodes represent genes
- edges represent regulatory interactions
- node features encode expression-derived statistics

### Operations

- The script internally performs k-fold cross-validation over the provided random subsets
- All k folds are trained in a single run
- One trained GAT model (.pt) is saved per fold

### Output

- One trained GAT model per fold (best_model_fold_X.pt)
- A CSV summary of fold-level performance metrics

## 2. Classifying a query dataset

### (scgrid_test.py)

This script applies a single trained GAT model to a query dataset and returns predicted cell-type labels.

### Input

- Path to a trained GAT model (.pt)
- A CTSGRN folder for the query dataset
- Node feature tables for the query dataset

### Operations

- The script classifies each CTSGRN independently
- A normalized entropy threshold can be used to assign "Unknown" labels
- The user must run this script once per trained model (per fold)

### Output

- A CSV file containing:
  1. predicted class labels
  2. per-class probabilities
  3. normalized entropy values

## 3. Ensemble majority voting

### (scgrid_majority_vote.py)

This script aggregates classification outputs across all k folds using majority voting.

### Input

- A directory containing all per-fold classification CSV files

### Operations

- Counts votes for each predicted class across folds
- Assigns the majority class if a unique winner exists
- Assigns "Unknown" in case of ties

### Output

A final CSV file containing:

- final predicted class per CTSGRN
- vote counts for each class



## State of the current repository

The current scGRID repository contains the core components required to evaluate the method itself, namely the Graph Attention Network (GAT) implementation and the ensemble majority-voting strategy used for classification. We are preparing an expanded and unified version of scGRID that will support the full workflow, callable from the command line.


## Notes

*This version is not a software release.
Scripts are currently configured via in-file parameters rather than a CLI.
Full pipeline automation and argument parsing are in currently in development.*

