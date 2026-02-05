# src/scgrid/preprocess.py
from __future__ import annotations
from pathlib import Path
import scanpy as sc


def convert_for_celloracle(
    input_h5ad: Path,
    output_h5ad: Path,
    annotation_col: str = "tabula_muris_annotation",
    embedding_layout: str = "fr",
) -> Path:
    """
    Reads an h5ad produced by your R preprocessing, adds:
      - neighbors + draw_graph embedding
      - adata.layers['raw_count']
      - adata.obs['custom_labels'] = adata.obs[annotation_col]
    Writes a CellOracle-ready h5ad.
    """
    input_h5ad = Path(input_h5ad)
    output_h5ad = Path(output_h5ad)
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(input_h5ad))

    # make sure categories are clean
    if annotation_col not in adata.obs:
        raise KeyError(f"Missing annotation column in adata.obs: {annotation_col}")

    # neighbors + graph layout
    sc.pp.neighbors(adata)
    sc.tl.draw_graph(adata, layout=embedding_layout)

    # raw counts for CellOracle
    adata.layers["raw_count"] = adata.X.copy()

    # custom labels for CellOracle
    if hasattr(adata.obs[annotation_col], "cat"):
        adata.obs[annotation_col] = adata.obs[annotation_col].cat.remove_unused_categories()

    adata.obs["custom_labels"] = adata.obs[annotation_col]

    adata.write(str(output_h5ad))
    return output_h5ad
