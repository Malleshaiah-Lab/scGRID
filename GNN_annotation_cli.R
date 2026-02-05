#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(ggplot2)
  library(scales)
  library(caret)
})

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- which(args == flag)
  if (length(idx) == 0) return(default)
  if (idx == length(args)) stop(paste("Missing value for", flag))
  args[idx + 1]
}

seurat_rds  <- get_arg("--seurat-rds")
pred_csv    <- get_arg("--pred-csv")
out_dir     <- get_arg("--out-dir")
truth_col   <- get_arg("--truth-col", "tabula_muris_annotation")
unk_label   <- get_arg("--unknown-label", "Unknown")
reduction   <- get_arg("--reduction", "umap")

if (is.null(seurat_rds) || is.null(pred_csv) || is.null(out_dir)) {
  stop("Usage: Rscript GNN_annotation_cli.R --seurat-rds <file.rds> --pred-csv <majority_vote_results.csv> --out-dir <dir> [--truth-col ...] [--unknown-label ...] [--reduction ...]")
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(seurat_rds)
mapping <- read.csv(pred_csv, stringsAsFactors = FALSE)

if (!truth_col %in% colnames(obj@meta.data)) {
  stop(paste0("truth_col '", truth_col, "' not found in Seurat meta.data. Available: ", paste(colnames(obj@meta.data), collapse=", ")))
}

# Build mapping: named vector Predicted_Class by CTSGRN_File (which equals the observed cluster/cell-type label)
map_vec <- setNames(mapping$Predicted_Class, mapping$CTSGRN_File)

obs <- as.character(obj@meta.data[[truth_col]])
pred <- map_vec[obs]
pred[is.na(pred)] <- unk_label

# Store prediction
all_levels <- c(sort(unique(obs)), unk_label)
obj@meta.data[["predicted_class"]] <- factor(pred, levels = all_levels)

# Prediction status
obj@meta.data[["Prediction_Status"]] <- ifelse(
  pred == unk_label, "unknown",
  ifelse(pred == obs, "correct", "incorrect")
)
obj@meta.data[["Prediction_Status"]] <- factor(obj@meta.data[["Prediction_Status"]],
                                              levels = c("correct","incorrect","unknown"))

# Palettes
obs_levels <- sort(unique(obs))
obs_pal <- hue_pal()(length(obs_levels))
names(obs_pal) <- obs_levels
pred_pal <- c(obs_pal, setNames("#AFB3B0", unk_label))

status_pal <- c(correct="green", incorrect="red", unknown="grey")

# UMAP PDFs
pdf(file.path(out_dir, paste0("umap_observed_", truth_col, ".pdf")), width=10, height=10)
print(DimPlot(obj, reduction=reduction, group.by=truth_col, cols=obs_pal))
dev.off()

pdf(file.path(out_dir, "umap_predicted_class.pdf"), width=10, height=10)
print(DimPlot(obj, reduction=reduction, group.by="predicted_class", cols=pred_pal))
dev.off()

pdf(file.path(out_dir, "umap_prediction_status.pdf"), width=10, height=10)
print(DimPlot(obj, reduction=reduction, group.by="Prediction_Status", cols=status_pal))
dev.off()

# Confusion matrix with Unknown column
observed_raw  <- obs
predicted_raw <- as.character(obj@meta.data[["predicted_class"]])

obs_levels <- sort(unique(observed_raw))
pred_levels <- c(obs_levels, unk_label)

observed  <- factor(observed_raw,  levels = obs_levels)
predicted <- factor(predicted_raw, levels = pred_levels)

cm <- table(Observed=observed, Predicted=predicted)
cm <- cm[obs_levels, pred_levels, drop=FALSE]

write.csv(as.matrix(cm), file.path(out_dir, "confusion_matrix_raw_counts.csv"), row.names=TRUE)

row_sums <- rowSums(cm)
cm_norm <- sweep(cm, 1, row_sums, FUN="/")
cm_norm[is.na(cm_norm)] <- 0
write.csv(as.matrix(cm_norm), file.path(out_dir, "confusion_matrix_proportions.csv"), row.names=TRUE)

# Overall accuracy including Unknown column (Unknown is never correct)
overlap <- intersect(colnames(cm), rownames(cm))
correct <- sum(diag(cm[overlap, overlap, drop=FALSE]))
total   <- sum(cm)
acc_incl_unknown <- correct / total

# caret stats (exclude Unknown by turning it into NA)
df_stats <- data.frame(
  Observed  = factor(observed_raw,  levels = obs_levels),
  Predicted = factor(predicted_raw, levels = obs_levels)
)
cm_stats <- confusionMatrix(data=df_stats$Predicted, reference=df_stats$Observed)

# Export caret stats
write.csv(as.matrix(cm_stats[["byClass"]]),
          file.path(out_dir, "confusion_matrix_stats_by_class.csv"),
          row.names=TRUE)

overall_stats <- cm_stats[["overall"]]
overall_stats["Overall Accuracy (incl Unknown column)"] <- acc_incl_unknown
write.csv(as.matrix(overall_stats),
          file.path(out_dir, "confusion_matrix_stats_overall.csv"),
          row.names=TRUE)

# Unknown stats
unknown_count <- sum(predicted_raw == unk_label)
unknown_prop  <- unknown_count / length(predicted_raw)
unknown_stats <- data.frame(
  Metric = c("Unknown Cell Count", "Unknown Cell Proportion"),
  Value  = c(unknown_count, unknown_prop)
)
write.csv(unknown_stats, file.path(out_dir, "unknown_stats.csv"), row.names=FALSE)

# Save updated Seurat object (with predicted_class + status)
saveRDS(obj, file.path(out_dir, "query_with_predictions.rds"))
