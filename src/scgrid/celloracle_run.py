# src/scgrid/celloracle_run.py
from __future__ import annotations
from pathlib import Path
import scanpy as sc
import pandas as pd
import celloracle as co


def run_celloracle_ctsgrns(
    h5ad_path: Path,
    out_dir: Path,
    metadata_column: str = "custom_labels",
    embedding_column: str = "X_draw_graph_fr",
    alpha: float = 10.0,
    p_filter: float = 0.001,
    threshold_number: int = 10000,
    n_cells_downsample: int = 30000,
    n_jobs: int = 4,
) -> Path:
    """
    Produces CTSGRN_<cluster>.csv files in out_dir, using CellOracle.
    Returns out_dir.
    """
    h5ad_path = Path(h5ad_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(h5ad_path))

    if metadata_column not in adata.obs:
        raise KeyError(f"Missing '{metadata_column}' in adata.obs")

    # ensure categorical, because your old script used .cat.categories
    if not hasattr(adata.obs[metadata_column], "cat"):
        adata.obs[metadata_column] = adata.obs[metadata_column].astype("category")

    # downsample to control runtime
    if adata.n_obs > n_cells_downsample:
        sc.pp.subsample(adata, n_obs=n_cells_downsample, random_state=123)

    # base GRN (mouse)
    base_grn = co.data.load_mouse_scATAC_atlas_base_GRN()

    oracle = co.Oracle()

    # CellOracle expects raw counts in X
    if "raw_count" not in adata.layers:
        raise KeyError("Expected adata.layers['raw_count'] (create it in preprocess step).")
    adata.X = adata.layers["raw_count"].copy()

    oracle.import_anndata_as_raw_count(
        adata=adata,
        cluster_column_name=metadata_column,
        embedding_name=embedding_column,
    )
    oracle.import_TF_data(TF_info_matrix=base_grn)

    adata_oracle = oracle.adata  # keep oracle's internal AnnData

    # CellOracle knn_imputation expects this layer name
    # (use normalized, non-log counts as the layer)
    sc.pp.normalize_total(adata_oracle, target_sum=1e4)
    adata_oracle.layers["normalized_count"] = adata_oracle.X.copy()

    # log1p goes to X (standard scanpy convention)
    sc.pp.log1p(adata_oracle)

    # HVGs (avoid skmisc requirement if you want)
    try:
        sc.pp.highly_variable_genes(adata_oracle, n_top_genes=2000, flavor="seurat_v3")
    except ImportError:
        print("[scGRID] skmisc not found; using cell_ranger HVGs")
        sc.pp.highly_variable_genes(adata_oracle, n_top_genes=2000, flavor="cell_ranger")

    adata_oracle = adata_oracle[:, adata_oracle.var["highly_variable"]].copy()

    # IMPORTANT: after subsetting, keep it as oracle.adata
    oracle.adata = adata_oracle

    # Ensure normalized_count survived slicing (it usually does, but safe guard)
    if "normalized_count" not in oracle.adata.layers:
        oracle.adata.layers["normalized_count"] = oracle.adata.X.copy()

    # PCA before knn_imputation
    sc.pp.scale(oracle.adata, max_value=10)
    sc.tl.pca(oracle.adata, n_comps=50)
    oracle.pcs = oracle.adata.obsm["X_pca"]

    # KNN parameters
    n_cell = oracle.adata.n_obs
    n_comps = min(oracle.pcs.shape[1], 50)

    k = max(10, int(0.025 * n_cell))
    k = min(k, n_cell - 1)

    oracle.knn_imputation(
        n_pca_dims=n_comps,
        k=k,
        balanced=True,
        b_sight=k * 8,
        b_maxl=k * 4,
        n_jobs=n_jobs,
    )

    # build links
    links = oracle.get_links(
        cluster_name_for_GRN_unit=metadata_column,
        alpha=alpha,
        verbose_level=1,
    )

    # filter + compute network scores
    links.filter_links(p=p_filter, weight="coef_abs", threshold_number=threshold_number)
    #links.get_network_score() # JL_COMMENT

    # export all clusters
    clusters = oracle.adata.obs[metadata_column].cat.categories.to_list()
    for cs in clusters:
        df = links.filtered_links.get(cs, None)
        if df is None or len(df) == 0:
            continue
        df.to_csv(out_dir / f"CTSGRN_{cs}.csv", index=False)

    return out_dir
