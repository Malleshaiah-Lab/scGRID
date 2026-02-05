# src/scgrid/train.py
from __future__ import annotations
from dataclasses import dataclass
import re
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.metrics import precision_recall_fscore_support


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


@dataclass
class TrainConfig:
    model_selected: str = "GAT"   # "GCN" or "GAT"
    conv1_channel: int = 64
    conv2_channel: int = 128
    gat_heads: int = 4

    features_to_remove: list[int] | None = None

    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 100
    batch_size: int = 32
    enable_early_stopping: bool = False
    patience: int = 15

    enable_significance_filter: bool = False
    significance_threshold: float = 1e-5


# ---- your existing helpers (slightly adapted to accept config) ----

def load_gene_features(feat_folder: Path):
    avg_expr_df    = pd.read_csv(feat_folder / "average_gene_expression_per_cell_type.csv", index_col=0)
    fold_change_df = pd.read_csv(feat_folder / "fold_change_per_gene_per_cell_type.csv", index_col=0)
    pct1_df        = pd.read_csv(feat_folder / "pct1_per_gene_per_cell_type.csv", index_col=0)
    pct2_df        = pd.read_csv(feat_folder / "pct2_per_gene_per_cell_type.csv", index_col=0)
    return avg_expr_df, fold_change_df, pct1_df, pct2_df


def ablate_features(data_list, features_to_remove):
    if features_to_remove is not None and isinstance(features_to_remove, int):
        features_to_remove = [features_to_remove]
    if features_to_remove is None or (isinstance(features_to_remove, list) and len(features_to_remove) == 0):
        return data_list

    num_features = data_list[0].x.size(1)
    keep_features = [i for i in range(num_features) if i not in features_to_remove]

    if len(keep_features) == 0:
        for data in data_list:
            data.x = torch.ones((data.x.size(0), 1))
    else:
        for data in data_list:
            data.x = data.x[:, keep_features]
    return data_list


def map_identifiers_to_ints(adj_matrix, cell_type, avg_expr_df, fold_change_df, pct1_df, pct2_df,
                            graph_labels, cfg: TrainConfig):
    if cfg.enable_significance_filter and "p" in adj_matrix.columns:
        adj_matrix = adj_matrix[adj_matrix["p"] < cfg.significance_threshold].copy()

    unique_genes = pd.concat([adj_matrix["source"], adj_matrix["target"]]).unique()
    gene_to_int = {gene: i for i, gene in enumerate(unique_genes)}
    int_to_gene = {i: gene for gene, i in gene_to_int.items()}

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
            node_features.append([0.0, 0.0, 0.0, 0.0])

    adj_matrix["source"] = adj_matrix["source"].map(gene_to_int)
    adj_matrix["target"] = adj_matrix["target"].map(gene_to_int)

    edge_index = torch.tensor([adj_matrix["source"].values, adj_matrix["target"].values], dtype=torch.long)
    edge_attr  = torch.tensor(adj_matrix["coef_mean"].values, dtype=torch.float)
    x          = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
    data.y = torch.tensor([graph_labels[cell_type]], dtype=torch.long)
    data.node_names = [int_to_gene[i] for i in range(len(int_to_gene))]
    return data


def load_ctsgrn_folder(adj_folder: Path, feat_folder: Path, output_classes, graph_labels, cfg: TrainConfig):
    avg_expr_df, fold_change_df, pct1_df, pct2_df = load_gene_features(feat_folder)
    data_list = []

    for ctype in output_classes:
        csv_file = adj_folder / f"CTSGRN_{ctype}.csv"
        if not csv_file.exists():
            continue

        adj_matrix = pd.read_csv(csv_file)
        data_obj = map_identifiers_to_ints(
            adj_matrix, ctype,
            avg_expr_df, fold_change_df, pct1_df, pct2_df,
            graph_labels, cfg
        )
        data_list.append(data_obj)

    return data_list


class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, conv1=16, conv2=32, drop=0.0):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, conv1)
        self.conv2 = GCNConv(conv1, conv2)
        self.fc = torch.nn.Linear(conv2, num_classes)
        self.dropout = drop

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class GATGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, conv1=16, conv2=32, heads=4, drop=0.0):
        super().__init__()
        self.conv1 = GATConv(num_node_features, conv1, heads=heads, concat=True)
        self.conv2 = GATConv(conv1 * heads, conv2, heads=1, concat=False)
        self.fc = torch.nn.Linear(conv2, num_classes)
        self.dropout = drop

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate_model(model, loader):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()
            preds_all.append(out.argmax(dim=1).cpu().numpy())
            labels_all.append(data.y.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    accuracy = (preds_all == labels_all).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / len(loader), accuracy, precision, recall, f1


def train_kfold(
    adj_folders: list[Path],
    feat_folders: list[Path],
    out_dir: Path,
    output_classes: list[str],
    cfg: TrainConfig,
):
    """
    Leave-one-folder-out CV like your original script.
    Saves best_model_fold_*.pt into out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_labels = {cell_type: i for i, cell_type in enumerate(output_classes)}

    all_data_by_folder = []
    for adj_folder, feat_folder in zip(adj_folders, feat_folders):
        all_data_by_folder.append(
            load_ctsgrn_folder(adj_folder, feat_folder, output_classes, graph_labels, cfg)
        )

    test_accuracies, test_f1s = [], []

    for fold_index in range(len(adj_folders)):
        train_data, test_data = [], []
        for i, data_list in enumerate(all_data_by_folder):
            if i == fold_index:
                test_data = [copy.deepcopy(d) for d in data_list]
            else:
                train_data.extend(copy.deepcopy(d) for d in data_list)

        train_data = ablate_features(train_data, cfg.features_to_remove)
        test_data  = ablate_features(test_data,  cfg.features_to_remove)

        train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
        test_loader  = DataLoader(test_data,  batch_size=cfg.batch_size, shuffle=False)

        num_node_features = train_data[0].num_node_features
        num_classes = len(graph_labels)

        if cfg.model_selected == "GCN":
            model = GCNGraphClassifier(num_node_features, num_classes,
                                       conv1=cfg.conv1_channel, conv2=cfg.conv2_channel, drop=cfg.dropout)
        else:
            model = GATGraphClassifier(num_node_features, num_classes,
                                       conv1=cfg.conv1_channel, conv2=cfg.conv2_channel,
                                       heads=cfg.gat_heads, drop=cfg.dropout)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        best_acc = 0.0
        best_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(cfg.num_epochs):
            train_one_epoch(model, train_loader, optimizer)
            _, te_acc, _, _, te_f1 = evaluate_model(model, test_loader)

            if te_acc > best_acc:
                best_acc = te_acc
                best_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if cfg.enable_early_stopping and epochs_no_improve >= cfg.patience:
                break

        model.load_state_dict(best_wts)
        _, final_acc, _, _, final_f1 = evaluate_model(model, test_loader)

        torch.save(model.state_dict(), out_dir / f"best_model_fold_{fold_index+1}.pt")
        test_accuracies.append(final_acc)
        test_f1s.append(final_f1)

    pd.DataFrame({
        "Fold": list(range(1, len(adj_folders)+1)),
        "TestAccuracy": test_accuracies,
        "TestF1": test_f1s,
    }).to_csv(out_dir / "kfold_results.csv", index=False)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj-folders", nargs="+", required=True)
    ap.add_argument("--feat-folders", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="GAT", choices=["GAT", "GCN"])
    args = ap.parse_args(argv)

    adj_folders = [Path(p) for p in args.adj_folders]
    output_classes = intersect_classes(adj_folders)
    if not output_classes:
        raise RuntimeError("Could not infer output classes from CTSGRN_*.csv in adj-folders.")
    
    cfg = TrainConfig(model_selected=args.model)
    train_kfold(
        [Path(p) for p in args.adj_folders],
        [Path(p) for p in args.feat_folders],
        Path(args.out_dir),
        output_classes=output_classes,
        cfg=cfg
    )
