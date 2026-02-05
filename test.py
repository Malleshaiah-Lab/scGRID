# src/scgrid/test.py
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


def load_gene_features(feat_folder: Path):
    avg_expr_df    = pd.read_csv(feat_folder / "average_gene_expression_per_cell_type.csv", index_col=0)
    fold_change_df = pd.read_csv(feat_folder / "fold_change_per_gene_per_cell_type.csv", index_col=0)
    pct1_df        = pd.read_csv(feat_folder / "pct1_per_gene_per_cell_type.csv", index_col=0)
    pct2_df        = pd.read_csv(feat_folder / "pct2_per_gene_per_cell_type.csv", index_col=0)
    return avg_expr_df, fold_change_df, pct1_df, pct2_df


def compute_normalized_entropy(prob_vector):
    eps = 1e-9
    k = len(prob_vector)
    h = 0.0
    for p in prob_vector:
        if p > 0:
            h -= p * math.log(p + eps)
    return h / math.log(k + eps)


def ablate_node_features(data: Data, features_to_remove):
    if isinstance(features_to_remove, int):
        features_to_remove = [features_to_remove]
    if not features_to_remove:
        return data

    keep = [i for i in range(data.x.size(1)) if i not in features_to_remove]
    if len(keep) == 0:
        data.x = torch.ones((data.num_nodes, 1))
    else:
        data.x = data.x[:, keep]
    return data


def build_graph(csv_path: Path, cell_type: str,
                avg_expr_df, fold_change_df, pct1_df, pct2_df,
                enable_significance_filter=False, sig_threshold=1e-5):
    adj_matrix = pd.read_csv(csv_path)

    if enable_significance_filter and "p" in adj_matrix.columns:
        adj_matrix = adj_matrix[adj_matrix["p"] < sig_threshold].copy()

    unique_genes = pd.concat([adj_matrix["source"], adj_matrix["target"]]).unique()
    gene_to_int = {g: i for i, g in enumerate(unique_genes)}

    node_features = []
    for gene in unique_genes:
        if gene in avg_expr_df.index:
            node_features.append([
                float(avg_expr_df.at[gene, cell_type]),
                float(fold_change_df.at[gene, cell_type]),
                float(pct1_df.at[gene, cell_type]),
                float(pct2_df.at[gene, cell_type]),
            ])
        else:
            node_features.append([0.0,0.0,0.0,0.0])

    adj_matrix["source"] = adj_matrix["source"].map(gene_to_int)
    adj_matrix["target"] = adj_matrix["target"].map(gene_to_int)

    edge_index = torch.tensor([adj_matrix["source"].values, adj_matrix["target"].values], dtype=torch.long)
    edge_attr  = torch.tensor(adj_matrix["coef_mean"].values, dtype=torch.float)
    x          = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
    return data


class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, conv1=64, conv2=128, drop=0.1):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, conv1)
        self.conv2 = GCNConv(conv1, conv2)
        self.fc = torch.nn.Linear(conv2, num_classes)
        self.drop = drop

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GATGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, conv1=64, conv2=128, heads=4, drop=0.1):
        super().__init__()
        self.conv1 = GATConv(num_node_features, conv1, heads=heads, concat=True)
        self.conv2 = GATConv(conv1 * heads, conv2, heads=1, concat=False)
        self.fc = torch.nn.Linear(conv2, num_classes)
        self.drop = drop

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def predict_with_model(
    model_path: Path,
    ctsgrn_dir: Path,
    feat_dir: Path,
    out_csv: Path,
    output_classes: list[str],
    model_type: str = "GAT",
    conv1: int = 64,
    conv2: int = 128,
    heads: int = 4,
    dropout: float = 0.1,
    entropy_threshold: float = 0.9,
    features_to_remove: list[int] | None = None,
):
    if not output_classes:
        raise ValueError("output_classes must be provided and non-empty.")
    
    graph_labels = {c: i for i, c in enumerate(output_classes)}
    reverse_labels = {v: k for k, v in graph_labels.items()}

    model_path = Path(model_path)
    ctsgrn_dir = Path(ctsgrn_dir)
    feat_dir   = Path(feat_dir)
    out_csv    = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    avg_expr_df, fold_change_df, pct1_df, pct2_df = load_gene_features(feat_dir)

    num_node_features = 4
    if features_to_remove:
        num_node_features = max(4 - len(features_to_remove), 1)

    if model_type == "GCN":
        model = GCNGraphClassifier(num_node_features, len(output_classes), conv1=conv1, conv2=conv2, drop=dropout)
    else:
        model = GATGraphClassifier(num_node_features, len(output_classes), conv1=conv1, conv2=conv2, heads=heads, drop=dropout)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    rows = []
    for ctype in output_classes:
        csv_path = ctsgrn_dir / f"CTSGRN_{ctype}.csv"
        if not csv_path.exists():
            continue

        data = build_graph(csv_path, ctype, avg_expr_df, fold_change_df, pct1_df, pct2_df)
        data = ablate_node_features(data, features_to_remove or [])
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

        with torch.no_grad():
            out = model(data)
            probs = torch.exp(out)[0].cpu().numpy()
            ne = compute_normalized_entropy(probs)
            pred_idx = int(out.argmax(dim=1).item())

        pred_class = "Unknown" if ne > entropy_threshold else reverse_labels[pred_idx]

        row = {"CTSGRN_File": ctype, "Predicted_Class": pred_class, "NormalizedEntropy": float(ne)}
        for i, cls in enumerate(output_classes):
            row[f"Prob_{cls}"] = float(probs[i])
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
