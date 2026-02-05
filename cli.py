# src/scgrid/cli.py

"""scGRID command-line interface (CLI).

This module defines the entry-point used by the `scgrid` command (or when calling
`python -m scgrid.cli`). It is intentionally thin: it only
  1) parses user inputs (paths + options),
  2) packages optional overrides for the R preprocessing step into `RPreprocessArgs`, and
  3) dispatches to the high-level workflows in `scgrid.workflow`:
     - `train_pipeline(...)` for model training
     - `classify_pipeline(...)` for inference using existing trained models

Inputs (high level)
-------------------
- --data-root: working/output root directory for scGRID runs.
- --input: either an existing Seurat .rds file, or a raw scRNA-seq input (e.g., 10X folder / CSV).
- --input-type: optional; if omitted, the workflow may infer it from the input path.

Outputs (high level)
--------------------
Both subcommands return a `run_dir` (printed at the end) where the workflow writes outputs.
The exact directory structure and produced files are defined in `scgrid.workflow`.

Note
----
This file is kept light on logic on purpose: most behavior should live in the pipelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scgrid.workflow import train_pipeline, classify_pipeline, RPreprocessArgs


def main(argv=None):
    ap = argparse.ArgumentParser("scGRID")

    ap.add_argument("--data-root", required=True, help="Separate data folder, e.g. /home/.../scgrid-data")
    ap.add_argument("--input", required=True, help="Path to scRNA-seq input: .rds OR raw (10X folder / CSV)")
    ap.add_argument("--input-type", default=None, choices=["rds", "raw"], help="Optional (else inferred)")
    ap.add_argument("--run-id", default=None, help="Optional run id (default uses timestamp)")

    # Common R args (optional overrides)
    ap.add_argument("--dataset", default=None, help="Output dataset name (default derived from tissue or filename)")
    ap.add_argument("--group-by", default=None, help="Metadata column for identities (default in R)")
    ap.add_argument("--assay", default=None, help="Assay name (default in R)")

    # Raw-only
    ap.add_argument("--method", default=None, choices=["Droplet", "FACS"], help="For --input-type raw")
    ap.add_argument("--tissue", default=None, help="For Tabula Muris annotation mapping (optional)")
    ap.add_argument("--annot-csv", default=None, help="TM annotation CSV (optional)")

    # Marker params (optional)
    ap.add_argument("--min-pct", type=float, default=None)
    ap.add_argument("--logfc-threshold", type=float, default=None)
    ap.add_argument("--only-pos", type=str, default=None, choices=["TRUE", "FALSE", "true", "false"])

    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--do-subsample", type=str, default=None, choices=["TRUE", "FALSE", "true", "false"])
    tr.add_argument("--num-subsets", type=int, default=None)
    tr.add_argument("--sampling-rate", type=float, default=None)
    tr.add_argument("--subset-suffix", default=None)
    tr.add_argument("--base-seed", type=int, default=None)

    cl = sub.add_parser("classify")
    cl.add_argument("--models-dir", required=True, help="Folder of trained .pt models")
    cl.add_argument("--entropy-threshold", type=float, default=0.9, help="Normalized entropy threshold above which predictions are set to 'Unknown' (default: 0.9).")

    args = ap.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    input_path = Path(args.input).resolve()

    def parse_opt_bool(s: str | None):
        if s is None:
            return None
        return s.lower() in ("true", "t", "1", "yes", "y")

    r_args = RPreprocessArgs(
        dataset=args.dataset,
        group_by=args.group_by,
        assay=args.assay,

        method=args.method,
        tissue=args.tissue,
        annot_csv=args.annot_csv,

        min_pct=args.min_pct,
        logfc_threshold=args.logfc_threshold,
        only_pos=parse_opt_bool(args.only_pos),
    )

    if args.cmd == "train":
        r_args.do_subsample = parse_opt_bool(args.do_subsample)
        r_args.num_subsets = args.num_subsets
        r_args.sampling_rate = args.sampling_rate
        r_args.subset_suffix = args.subset_suffix
        r_args.base_seed = args.base_seed

        run_dir = train_pipeline(
            data_root=data_root,
            input_path=input_path,
            input_type=args.input_type,
            run_id=args.run_id,
            r_args=r_args,
        )
    else:
        run_dir = classify_pipeline(
            data_root=data_root,
            input_path=input_path,
            input_type=args.input_type,
            models_dir=Path(args.models_dir).resolve(),
            run_id=args.run_id,
            r_args=r_args,
            entropy_threshold=args.entropy_threshold,
        )

    print(f"\nDONE. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()
