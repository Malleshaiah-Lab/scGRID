# src/scgrid/workflow.py
from __future__ import annotations

import datetime
import os
import subprocess
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional, Dict, Any, List

from scgrid.preprocess import convert_for_celloracle
from scgrid.celloracle_run import run_celloracle_ctsgrns
from scgrid.train import TrainConfig, train_kfold
from scgrid.test import predict_with_model
from scgrid.majority_vote import majority_vote


@dataclass
class RPreprocessArgs:
    # core
    dataset: Optional[str] = None
    group_by: Optional[str] = None
    assay: Optional[str] = None

    # raw-only
    method: Optional[str] = None          # Droplet | FACS
    tissue: Optional[str] = None
    annot_csv: Optional[str] = None

    # subsampling
    do_subsample: Optional[bool] = None
    num_subsets: Optional[int] = None
    sampling_rate: Optional[float] = None
    subset_suffix: Optional[str] = None
    base_seed: Optional[int] = None

    # marker params
    min_pct: Optional[float] = None
    logfc_threshold: Optional[float] = None
    only_pos: Optional[bool] = None


def make_run_dir(data_root: Path, run_id: Optional[str]) -> Path:
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(data_root) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _bool_to_str(x: bool) -> str:
    return "TRUE" if x else "FALSE"


def _build_r_flags(args: RPreprocessArgs) -> List[str]:
    """
    Convert RPreprocessArgs to CLI flags expected by the rewritten scGRID_preprocessing.R.
    Only includes flags that are not None.
    """
    flags: List[str] = []

    def add(flag: str, val: Any):
        if val is None:
            return
        if isinstance(val, bool):
            flags.extend([flag, _bool_to_str(val)])
        else:
            flags.extend([flag, str(val)])

    add("--dataset", args.dataset)
    add("--group-by", args.group_by)
    add("--assay", args.assay)

    add("--method", args.method)
    add("--tissue", args.tissue)
    add("--annot-csv", args.annot_csv)

    add("--do-subsample", args.do_subsample)
    add("--num-subsets", args.num_subsets)
    add("--sampling-rate", args.sampling_rate)
    add("--subset-suffix", args.subset_suffix)
    add("--base-seed", args.base_seed)

    add("--min-pct", args.min_pct)
    add("--logfc-threshold", args.logfc_threshold)
    add("--only-pos", args.only_pos)

    return flags


def run_r_preprocess(
    *,
    mode: str,
    out_dir: Path,
    input_path: Path,
    input_type: Optional[str],
    r_args: RPreprocessArgs,
) -> None:
    """
    Calls packaged scGRID_preprocessing.R using Rscript.

    Expects scGRID_preprocessing.R to accept:
      --mode, --out-dir, --input, [--input-type],
      plus optional flags (method/tissue/annot-csv/dataset/group-by/etc.)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rscript_bin = os.environ.get("SCGRID_RSCRIPT", "Rscript")

    r_trav = resources.files("scgrid").joinpath("r", "scGRID_preprocessing.R")
    with resources.as_file(r_trav) as r_script_path:
        cmd = [
            rscript_bin,
            str(r_script_path),
            "--mode", mode,
            "--out-dir", str(out_dir),
            "--input", str(input_path),
        ]
        if input_type:
            cmd += ["--input-type", input_type]

        cmd += _build_r_flags(r_args)

        subprocess.run(cmd, check=True)


def run_r_annotation(
    *,
    out_dir: Path,
    seurat_rds: Path,
    pred_csv: Path,
    truth_col: str = "tabula_muris_annotation",
    unknown_label: str = "Unknown",
    reduction: str = "umap",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rscript_bin = os.environ.get("SCGRID_RSCRIPT", "Rscript")

    r_trav = resources.files("scgrid").joinpath("r", "GNN_annotation_cli.R")
    with resources.as_file(r_trav) as r_script_path:
        cmd = [
            rscript_bin,
            str(r_script_path),
            "--seurat-rds", str(seurat_rds),
            "--pred-csv", str(pred_csv),
            "--out-dir", str(out_dir),
            "--truth-col", truth_col,
            "--unknown-label", unknown_label,
            "--reduction", reduction,
        ]
        subprocess.run(cmd, check=True)


def infer_output_classes_from_ctsgrns(ctsgrn_dir: Path) -> list[str]:
    classes = []
    for p in Path(ctsgrn_dir).glob("CTSGRN_*.csv"):
        m = re.match(r"CTSGRN_(.*)\.csv$", p.name)
        if m:
            classes.append(m.group(1))
    return sorted(set(classes))


def intersect_classes(adj_folders: list[Path]) -> list[str]:
    sets = [set(infer_output_classes_from_ctsgrns(f)) for f in adj_folders]
    return sorted(set.intersection(*sets)) if sets else []


def train_pipeline(
    *,
    data_root: Path,
    input_path: Path,
    input_type: Optional[str],
    run_id: Optional[str] = None,
    r_args: Optional[RPreprocessArgs] = None,
) -> Path:
    """
    Train mode:
      R preprocess -> subsets -> convert_for_celloracle -> CellOracle -> train_kfold
    """
    run_dir = make_run_dir(data_root, run_id)

    preproc_out = run_dir / "preprocessing_output"
    celloracle_out = run_dir / "celloracle_output" / "train_subsets"
    models_out = run_dir / "models"
    celloracle_out.mkdir(parents=True, exist_ok=True)
    models_out.mkdir(parents=True, exist_ok=True)

    if r_args is None:
        r_args = RPreprocessArgs()

    run_r_preprocess(
        mode="train",
        out_dir=preproc_out,
        input_path=input_path,
        input_type=input_type,
        r_args=r_args,
    )

    # Expected output from rewritten R script:
    # preproc_out/DATASET_NAME/subsets/<subset_name>/<subset_name>.h5ad
    dataset_dirs = sorted([p for p in preproc_out.iterdir() if p.is_dir()])
    if len(dataset_dirs) != 1:
        raise RuntimeError(f"Expected exactly 1 dataset folder under {preproc_out}, found {len(dataset_dirs)}")
    dataset_dir = dataset_dirs[0]

    subsets_root = dataset_dir / "subsets"
    if not subsets_root.exists():
        raise FileNotFoundError(f"Missing subsets folder: {subsets_root}")

    subset_dirs = sorted([p for p in subsets_root.iterdir() if p.is_dir()])
    if not subset_dirs:
        raise RuntimeError(f"No subset dirs found in: {subsets_root}")

    adj_folders: List[Path] = []
    feat_folders: List[Path] = []

    for subset in subset_dirs:
        subset_name = subset.name
        input_h5ad = subset / f"{subset_name}.h5ad"
        if not input_h5ad.exists():
            raise FileNotFoundError(f"Missing subset h5ad: {input_h5ad}")

        subset_celloracle_dir = celloracle_out / subset_name
        subset_celloracle_dir.mkdir(parents=True, exist_ok=True)

        converted_h5ad = subset_celloracle_dir / f"{subset_name}.celloracle_ready.h5ad"
        convert_for_celloracle(input_h5ad, converted_h5ad)

        run_celloracle_ctsgrns(converted_h5ad, subset_celloracle_dir)

        adj_folders.append(subset_celloracle_dir)
        feat_folders.append(subset)  # feature CSVs created by R script in subset folder

    cfg = TrainConfig(model_selected="GAT")

    output_classes = intersect_classes(adj_folders)
    if not output_classes:
        raise RuntimeError("Could not infer output classes from CTSGRN_*.csv across adj_folders.")


    train_kfold(
        adj_folders,
        feat_folders,
        models_out,
        output_classes=output_classes,
        cfg=cfg,
    )

    return run_dir


def classify_pipeline(
    *,
    data_root: Path,
    input_path: Path,
    input_type: Optional[str],
    models_dir: Path,
    run_id: Optional[str] = None,
    r_args: Optional[RPreprocessArgs] = None,
    entropy_threshold: float = 0.9,
) -> Path:
    """
    Classify mode:
      R preprocess -> original -> convert_for_celloracle -> CellOracle -> predict per model -> majority vote
    """
    run_dir = make_run_dir(data_root, run_id)

    preproc_out = run_dir / "preprocessing_output"
    celloracle_out = run_dir / "celloracle_output" / "query"
    preds_dir = run_dir / "predictions"
    fold_preds = preds_dir / "fold_preds"

    celloracle_out.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    fold_preds.mkdir(parents=True, exist_ok=True)

    if r_args is None:
        r_args = RPreprocessArgs()

    # classify should NOT subsample by default; only override if user asked explicitly
    if r_args.do_subsample is None:
        r_args.do_subsample = False

    run_r_preprocess(
        mode="classify",
        out_dir=preproc_out,
        input_path=input_path,
        input_type=input_type,
        r_args=r_args,
    )

    # Expected output from rewritten R script:
    # preproc_out/DATASET_NAME/original/DATASET_NAME.h5ad
    dataset_dirs = sorted([p for p in preproc_out.iterdir() if p.is_dir()])
    if len(dataset_dirs) != 1:
        raise RuntimeError(f"Expected exactly 1 dataset folder under {preproc_out}, found {len(dataset_dirs)}")
    dataset_dir = dataset_dirs[0]

    orig_dir = dataset_dir / "original"
    dataset_name = dataset_dir.name
    input_h5ad = orig_dir / f"{dataset_name}.h5ad"
    if not input_h5ad.exists():
        raise FileNotFoundError(f"Missing original h5ad: {input_h5ad}")

    converted_h5ad = celloracle_out / f"{dataset_name}.celloracle_ready.h5ad"
    convert_for_celloracle(input_h5ad, converted_h5ad)

    run_celloracle_ctsgrns(converted_h5ad, celloracle_out)
    output_classes = infer_output_classes_from_ctsgrns(celloracle_out)
    if not output_classes:
        raise RuntimeError("No CTSGRN_*.csv files found; cannot classify.")

    feat_dir = orig_dir  # feature CSVs created by R script in original folder

    model_paths = sorted(Path(models_dir).glob("*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No .pt models in: {models_dir}")

    for i, mp in enumerate(model_paths, start=1):
        out_csv = fold_preds / f"annotation_results_model_{i}.csv"
        predict_with_model(mp, ctsgrn_dir=celloracle_out, feat_dir=feat_dir, output_classes=output_classes, out_csv=out_csv, entropy_threshold=entropy_threshold,)

    final_csv = preds_dir / "majority_vote_results.csv"
    majority_vote(fold_preds, final_csv)

    # annotation report on the query Seurat object
    reports_dir = run_dir / "annotation_reports"

    # Prefer the preprocessed query .rds (so it has the same metadata/UMAP as what scGRID used)
    seurat_rds = orig_dir / f"{dataset_name}.rds"
    if seurat_rds.exists():
        truth_col = (r_args.group_by or "tabula_muris_annotation")
        run_r_annotation(
            out_dir=reports_dir,
            seurat_rds=seurat_rds,
            pred_csv=final_csv,
            truth_col=truth_col,
            unknown_label="Unknown",
            reduction="umap",
        )
    else:
        print(f"[WARN] Seurat RDS not found at {seurat_rds}; skipping R annotation report.")

    return run_dir
