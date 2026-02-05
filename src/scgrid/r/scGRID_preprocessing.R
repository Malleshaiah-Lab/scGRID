#!/usr/bin/env Rscript

# ============================================================
# scGRID - Preprocessing (argument-driven)
#
# Inputs:
#   --input-type rds --input <seurat.rds>
#   OR
#   --input-type raw --method Droplet --input <10x_folder>
#   OR
#   --input-type raw --method FACS    --input <counts.csv>
#
# Outputs (under --out-dir / --dataset):
#   original/
#     <dataset>.rds
#     <dataset>.h5ad
#     average_gene_expression_per_cell_type.csv
#     fold_change_per_gene_per_cell_type.csv
#     pct1_per_gene_per_cell_type.csv
#     pct2_per_gene_per_cell_type.csv
#
#   subsets/ (train mode + subsampling enabled)
#     <dataset><subset-suffix><i>/
#       <subset_name>.rds
#       <subset_name>.h5ad
#       node feature csvs...
# ============================================================

suppressPackageStartupMessages({
  library(Seurat)
  library(dplyr)
  library(stringr)
  library(SingleCellExperiment)
  library(zellkonverter)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  if (idx == length(args)) return(default)
  return(args[idx + 1])
}

has_flag <- function(flag) {
  any(args == flag)
}

parse_bool <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  x <- tolower(as.character(x))
  if (x %in% c("true", "t", "1", "yes", "y")) return(TRUE)
  if (x %in% c("false", "f", "0", "no", "n")) return(FALSE)
  return(default)
}

die <- function(...) {
  msg <- paste0(...)
  stop(msg, call. = FALSE)
}

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# -----------------------------
# Parse CLI args
# -----------------------------

MODE <- get_arg("--mode", "train")  # train | classify
if (!(MODE %in% c("train", "classify"))) {
  die("Invalid --mode: ", MODE, " (use train or classify)")
}

INPUT_TYPE <- get_arg("--input-type", NULL)  # rds | raw
INPUT_PATH <- get_arg("--input", NULL)
OUT_DIR <- get_arg("--out-dir", NULL)

if (is.null(INPUT_PATH)) die("Missing required arg: --input")
if (is.null(OUT_DIR))   die("Missing required arg: --out-dir")

INPUT_PATH <- normalizePath(INPUT_PATH, mustWork = TRUE)
OUT_DIR <- normalizePath(OUT_DIR, mustWork = FALSE)
ensure_dir(OUT_DIR)

# For raw input
METHOD <- get_arg("--method", "Droplet")     # Droplet | FACS
TISSUE <- get_arg("--tissue", NULL)          # optional, used for project name + TM mapping
ANNOT_CSV <- get_arg("--annot-csv", NULL)    # optional

# Dataset naming
DATASET_NAME <- get_arg("--dataset", NULL)
if (is.null(DATASET_NAME)) {
  # fallback: tissue if provided else basename(input)
  if (!is.null(TISSUE)) {
    DATASET_NAME <- TISSUE
  } else {
    b <- basename(INPUT_PATH)
    DATASET_NAME <- sub("\\.rds$", "", b)
    DATASET_NAME <- sub("\\.csv$", "", DATASET_NAME)
  }
}

# Seurat / features config
GROUP_BY <- get_arg("--group-by", "tabula_muris_annotation")
ASSAY_NAME <- get_arg("--assay", "RNA")

# Marker calling params
MIN_PCT <- as.numeric(get_arg("--min-pct", "0.25"))
LOGFC_THRESHOLD <- as.numeric(get_arg("--logfc-threshold", "0.25"))
ONLY_POS <- parse_bool(get_arg("--only-pos", "TRUE"), default = TRUE)

# Subsampling params
SAMPLING_RATE <- as.numeric(get_arg("--sampling-rate", "0.7"))
NUM_SUBSETS <- as.integer(get_arg("--num-subsets", "10"))
SUBSET_SUFFIX <- get_arg("--subset-suffix", "_RandSub")
BASE_SEED <- as.integer(get_arg("--base-seed", "42"))

# Subsampling enable:
# - classify: default FALSE
# - train:    default TRUE
DO_SUBSAMPLE_DEFAULT <- if (MODE == "train") TRUE else FALSE
DO_SUBSAMPLE <- parse_bool(get_arg("--do-subsample", NULL), default = DO_SUBSAMPLE_DEFAULT)

# Infer input-type if not provided
if (is.null(INPUT_TYPE)) {
  if (grepl("\\.rds$", INPUT_PATH, ignore.case = TRUE)) {
    INPUT_TYPE <- "rds"
  } else {
    INPUT_TYPE <- "raw"
  }
}
if (!(INPUT_TYPE %in% c("rds", "raw"))) {
  die("Invalid --input-type: ", INPUT_TYPE, " (use rds or raw)")
}

# -----------------------------
# Helper functions
# -----------------------------

safe_set_idents <- function(seurat_obj, group_by) {
  if (!is.null(group_by) && group_by %in% colnames(seurat_obj@meta.data)) {
    seurat_obj <- SetIdent(seurat_obj, value = seurat_obj@meta.data[[group_by]])
  }
  return(seurat_obj)
}

apply_tabula_muris_annotation <- function(seurat_obj, tissue, annot_csv) {
  if (is.null(annot_csv) || !file.exists(annot_csv)) {
    message("[scGRID_preprocess] No --annot-csv provided/found; skipping TM annotation mapping.")
    return(seurat_obj)
  }
  if (is.null(tissue)) {
    message("[scGRID_preprocess] --annot-csv provided, but --tissue is NULL; skipping TM mapping.")
    return(seurat_obj)
  }

  annotations <- read.csv(annot_csv)

  # tissue like "Marrow-10X_P7_3" -> split -> tissue_main="Marrow", tissue_sub="10X"
  parts <- str_split(tissue, "-")[[1]]
  tissue_main <- parts[1]
  tissue_sub  <- if (length(parts) >= 2) parts[2] else parts[1]

  annotations <- annotations[annotations$tissue == tissue_main, , drop = FALSE]
  annotations <- annotations[grepl(tissue_sub, annotations$cell), , drop = FALSE]

  idents <- annotations$cell_ontology_class
  name_idents <- annotations$cell

  # Convert "xxx_xxx_xxx_<BARCODE>_..." -> "<BARCODE>-1"
  new_name_idents <- vapply(name_idents, function(x) {
    p <- str_split(x, "_")[[1]]
    bc <- if (length(p) >= 4) p[4] else p[length(p)]
    paste0(bc, "-1")
  }, FUN.VALUE = character(1))

  names(idents) <- new_name_idents
  idents <- as.factor(idents)

  keep <- intersect(colnames(seurat_obj), names(idents))
  seurat_obj <- seurat_obj[, keep, drop = FALSE]

  seurat_obj@meta.data$tabula_muris_annotation <- idents[colnames(seurat_obj)]
  seurat_obj <- SetIdent(seurat_obj, value = seurat_obj@meta.data$tabula_muris_annotation)
  return(seurat_obj)
}

load_from_raw <- function(method, input_path, project_name = "scGRID") {
  if (method == "Droplet") {
    message("[scGRID_preprocess] Reading 10X from: ", input_path)
    counts <- Read10X(input_path)
  } else if (method == "FACS") {
    message("[scGRID_preprocess] Reading FACS CSV from: ", input_path)
    counts <- read.csv(input_path, row.names = 1, check.names = FALSE)
  } else {
    die("METHOD must be 'Droplet' or 'FACS'. Got: ", method)
  }
  seu <- CreateSeuratObject(counts = counts, project = project_name, min.cells = 0, min.features = 0)
  return(seu)
}

write_node_features <- function(seurat_obj, out_dir, group_by, assay_name) {
  ensure_dir(out_dir)

  gb <- NULL
  if (!is.null(group_by) && group_by %in% colnames(seurat_obj@meta.data)) gb <- group_by

  # Average expression
  avg <- AverageExpression(
    seurat_obj,
    group.by = if (is.null(gb)) NULL else gb,
    assays = assay_name
  )
  avg_mat <- as.data.frame(as.matrix(avg[[assay_name]]))
  write.csv(avg_mat, file = file.path(out_dir, "average_gene_expression_per_cell_type.csv"))

  # Markers -> fold-change + pct.1 + pct.2
  markers <- FindAllMarkers(
    seurat_obj,
    only.pos = ONLY_POS,
    min.pct = MIN_PCT,
    logfc.threshold = LOGFC_THRESHOLD,
    group.by = if (is.null(gb)) NULL else gb
  )

  # Seurat v4/v5: marker columns can be avg_log2FC or avg_logFC depending on settings
  fc_col <- if ("avg_log2FC" %in% colnames(markers)) "avg_log2FC" else if ("avg_logFC" %in% colnames(markers)) "avg_logFC" else NA
  if (is.na(fc_col)) die("Markers table missing avg_log2FC/avg_logFC column.")

  all_genes <- rownames(seurat_obj[[assay_name]])
  cell_types <- unique(markers$cluster)

  # helper to build matrices
  build_mat <- function(value_col) {
    mat <- matrix(0, nrow = length(all_genes), ncol = length(cell_types),
                  dimnames = list(all_genes, cell_types))
    for (ct in cell_types) {
      ct_markers <- markers %>% filter(cluster == ct)
      common <- intersect(ct_markers$gene, all_genes)
      mat[common, ct] <- ct_markers[[value_col]][match(common, ct_markers$gene)]
    }
    return(mat)
  }

  fc_mat <- build_mat(fc_col)
  write.csv(as.data.frame(fc_mat), file = file.path(out_dir, "fold_change_per_gene_per_cell_type.csv"))

  pct1_mat <- build_mat("pct.1")
  write.csv(as.data.frame(pct1_mat), file = file.path(out_dir, "pct1_per_gene_per_cell_type.csv"))

  pct2_mat <- build_mat("pct.2")
  write.csv(as.data.frame(pct2_mat), file = file.path(out_dir, "pct2_per_gene_per_cell_type.csv"))

  invisible(markers)
}

export_rds_h5ad <- function(seurat_obj, out_dir, dataset_name, assay_name, group_by) {
  ensure_dir(out_dir)

  # Save RDS
  rds_file <- file.path(out_dir, paste0(dataset_name, ".rds"))
  saveRDS(seurat_obj, rds_file)

  # Node features
  write_node_features(seurat_obj, out_dir, group_by, assay_name)

  # Convert to h5ad
  h5ad_file <- file.path(out_dir, paste0(dataset_name, ".h5ad"))
  sce <- as.SingleCellExperiment(seurat_obj, assay = assay_name)
  zellkonverter::writeH5AD(sce, file = h5ad_file)

  message("[scGRID_preprocess] Exported: ", rds_file)
  message("[scGRID_preprocess] Exported: ", h5ad_file)
}

make_subsampled_objects <- function(seurat_obj, sampling_rate, num_subsets, base_seed, group_by) {
  # Use group_by if present, else Idents
  if (!is.null(group_by) && group_by %in% colnames(seurat_obj@meta.data)) {
    seurat_obj <- SetIdent(seurat_obj, value = as.factor(seurat_obj@meta.data[[group_by]]))
  }
  idents <- Idents(seurat_obj)
  cells_by_ident <- split(x = colnames(seurat_obj), f = idents)

  subsets <- vector("list", length = num_subsets)

  for (i in seq_len(num_subsets)) {
    set.seed(base_seed + i)

    sampled_cells <- c()
    for (ident in names(cells_by_ident)) {
      cell_vec <- cells_by_ident[[ident]]
      n_to_sample <- max(1, round(sampling_rate * length(cell_vec)))
      sampled_cells <- c(sampled_cells, sample(cell_vec, size = n_to_sample))
    }

    subsets[[i]] <- subset(seurat_obj, cells = sampled_cells)
  }

  return(subsets)
}

# -----------------------------
# Main
# -----------------------------

message("[scGRID_preprocess] mode=", MODE,
        " input-type=", INPUT_TYPE,
        " dataset=", DATASET_NAME,
        " out-dir=", OUT_DIR)

# 1) Load / create Seurat
if (INPUT_TYPE == "raw") {
  if (is.null(METHOD)) die("For --input-type raw, you must provide --method Droplet|FACS")
  proj <- if (!is.null(TISSUE)) TISSUE else DATASET_NAME
  seurat_obj <- load_from_raw(METHOD, INPUT_PATH, project_name = proj)
  seurat_obj <- apply_tabula_muris_annotation(seurat_obj, TISSUE, ANNOT_CSV)
} else {
  # rds
  seurat_obj <- readRDS(INPUT_PATH)
}

# 2) Validate assay
if (!(ASSAY_NAME %in% names(seurat_obj@assays))) {
  die("ASSAY_NAME '", ASSAY_NAME, "' not found in Seurat object assays: ",
      paste(names(seurat_obj@assays), collapse = ", "))
}

# 3) Set Idents
seurat_obj <- safe_set_idents(seurat_obj, GROUP_BY)

# 4) Export original
orig_dir <- file.path(OUT_DIR, DATASET_NAME, "original")
export_rds_h5ad(seurat_obj, orig_dir, DATASET_NAME, ASSAY_NAME, GROUP_BY)

# 5) Export subsamples (train only by default)
if (DO_SUBSAMPLE) {
  subs <- make_subsampled_objects(seurat_obj, SAMPLING_RATE, NUM_SUBSETS, BASE_SEED, GROUP_BY)

  for (i in seq_along(subs)) {
    subset_name <- paste0(DATASET_NAME, SUBSET_SUFFIX, i)
    subset_dir  <- file.path(OUT_DIR, DATASET_NAME, "subsets", subset_name)
    export_rds_h5ad(subs[[i]], subset_dir, subset_name, ASSAY_NAME, GROUP_BY)
  }
} else {
  message("[scGRID_preprocess] Subsampling disabled.")
}

message("[scGRID_preprocess] Done.")

