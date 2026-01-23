import os
import matplotlib.pyplot as plt
import copy
import uuid
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from sklearn.metrics import precision_recall_fscore_support

WORKSPACE_PATH = "" # <path_to_workspace>
if not os.path.exists(WORKSPACE_PATH):
    os.makedirs(WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

######################
## 1. Global config ##
######################
model_selected = "GAT"  # "GAT" or "GCN"
conv1_channel = 64
conv2_channel = 128
gat_heads = 4

features_to_remove = None

cssgrn_folders = [] # <path_to_cssgrn_folder>
feature_folders = [] # <path_to_feature_folders>

output_classes = [
    'B cell', 'erythrocyte', 'Fraction A pre-pro B cell',
    'granulocyte', 'hematopoietic stem cell', 'macrophage',
    'monocyte', 'T cell'
]

graph_labels = {cell_type: i for i, cell_type in enumerate(output_classes)}

dropout = 0.1
learning_rate = 1e-3
weight_decay = 0.0
num_epochs = 100
batch_size = 32
enable_early_stopping = False
patience = 15
delta = 1e-4
enable_significance_filter = False
significance_threshold = 1e-5

###########################
## 2. Load gene features ##
###########################
def load_gene_features(avg_path, fold_path, pct1_path, pct2_path):
    avg_expr_df      = pd.read_csv(avg_path, index_col=0)
    fold_change_df   = pd.read_csv(fold_path, index_col=0)
    pct1_df          = pd.read_csv(pct1_path, index_col=0)
    pct2_df          = pd.read_csv(pct2_path, index_col=0)
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
            num_nodes = data.x.size(0)
            data.x = torch.ones((num_nodes, 1))
    else:
        for data in data_list:
            data.x = data.x[:, keep_features]
    return data_list

#####################################################
## 3. Map adjacency => PyG Data with node features ##
#####################################################
def map_identifiers_to_ints(adj_matrix, cell_type,
                            avg_expr_df, fold_change_df, pct1_df, pct2_df):
    if enable_significance_filter:
        adj_matrix = adj_matrix[adj_matrix['p'] < significance_threshold].copy()

    # Build gene↔int maps
    unique_genes = pd.concat([adj_matrix['source'], adj_matrix['target']]).unique()
    gene_to_int = {gene: i for i, gene in enumerate(unique_genes)}
    int_to_gene = {i: gene for gene, i in gene_to_int.items()}

    # Assemble node‐feature matrix
    node_features = []
    for gene in unique_genes:
        if gene in avg_expr_df.index:
            avg_val  = avg_expr_df.at[gene, cell_type]
            fold_val = fold_change_df.at[gene, cell_type]
            pct1_val = pct1_df.at[gene, cell_type]
            pct2_val = pct2_df.at[gene, cell_type]
            node_features.append([avg_val, fold_val, pct1_val, pct2_val])
        else:
            node_features.append([0.0, 0.0, 0.0, 0.0])

    # Remap edges
    adj_matrix['source'] = adj_matrix['source'].map(gene_to_int)
    adj_matrix['target'] = adj_matrix['target'].map(gene_to_int)

    edge_index = torch.tensor(
        [adj_matrix['source'].values, adj_matrix['target'].values],
        dtype=torch.long
    )
    edge_attr = torch.tensor(adj_matrix['coef_mean'].values, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )
    data.y = torch.tensor([graph_labels[cell_type]], dtype=torch.long)
    data.node_names = [int_to_gene[i] for i in range(len(int_to_gene))]
    return data

#################################################
## 4. Load all cell types from a single folder ##
#################################################
def load_cssgrn_folder(adj_folder, feat_folder, cell_types):
    avg_expr_df, fold_change_df, pct1_df, pct2_df = load_gene_features(
        os.path.join(feat_folder, "average_gene_expression_per_cell_type.csv"),
        os.path.join(feat_folder, "fold_change_per_gene_per_cell_type.csv"),
        os.path.join(feat_folder, "pct1_per_gene_per_cell_type.csv"),
        os.path.join(feat_folder, "pct2_per_gene_per_cell_type.csv"),
    )

    data_list = []
    for ctype in cell_types:
        csv_file = f"CSSGRN_{ctype}.csv"
        full_path = os.path.join(adj_folder, csv_file)
        if not os.path.exists(full_path):
            print(f"[WARNING] Missing file: {full_path}")
            continue

        adj_matrix = pd.read_csv(full_path)
        data_obj = map_identifiers_to_ints(
            adj_matrix, ctype,
            avg_expr_df, fold_change_df, pct1_df, pct2_df
        )
        data_list.append(data_obj)

    return data_list

################################
## 5. GNN models (GCN or GAT) ##
################################
class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, conv1=16, conv2=32, drop=0.0):
        super(GCNGraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, conv1)
        self.conv2 = GCNConv(conv1, conv2)
        self.fc = torch.nn.Linear(conv2, num_classes)
        self.dropout = drop

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GATGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, conv1=16, conv2=32, heads=4, drop=0.0):
        super(GATGraphClassifier, self).__init__()
        self.conv1 = GATConv(num_node_features, conv1, heads=heads, concat=True)
        self.conv2 = GATConv(conv1 * heads, conv2, heads=1, concat=False)
        self.fc = torch.nn.Linear(conv2, num_classes)
        self.dropout = drop

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

###############################
## 6. Train / Evaluate loops ##
###############################
def train_one_epoch(model, loader, optimizer):
    """
    Trains for a single epoch (for quick debugging).
    Returns the average loss and accuracy (no precision/recall here, just basic).
    """
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
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate_model(model, loader):
    """
    Computes average loss, accuracy, precision, recall, F1 (macro) for the entire loader.
    """
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = F.nll_loss(out, data.y)
            total_loss += loss.item()

            preds = out.argmax(dim=1).cpu().numpy()
            labs  = data.y.cpu().numpy()
            preds_all.append(preds)
            labels_all.append(labs)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    avg_loss = total_loss / len(loader)
    accuracy = (preds_all == labels_all).mean()

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all, preds_all, average='macro', zero_division=0
    )
    return avg_loss, accuracy, precision, recall, f1

#####################################################
## 7. Main "k-fold" with the random-subset folders ##
#####################################################
def run_kfold_experiment():
    test_accuracies  = []
    test_precisions  = []
    test_recalls     = []
    test_f1s         = []

    all_data_by_folder = []
    for adj_folder, feat_folder in zip(cssgrn_folders, feature_folders):
        data_list = load_cssgrn_folder(adj_folder, feat_folder, output_classes)
        all_data_by_folder.append(data_list)

    # "leave-one-out" across k folds
    for fold_index in range(len(cssgrn_folders)):
        train_data, test_data = [], []
        for i, data_list in enumerate(all_data_by_folder):
            if i == fold_index:
                test_data = [copy.deepcopy(d) for d in data_list]
            else:
                train_data.extend(copy.deepcopy(d) for d in data_list)

        train_data = ablate_features(train_data, features_to_remove)
        test_data  = ablate_features(test_data,  features_to_remove)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        num_node_features = train_data[0].num_node_features
        num_classes = len(graph_labels)

        if model_selected == "GCN":
            model = GCNGraphClassifier(
                num_node_features, num_classes,
                conv1=conv1_channel,
                conv2=conv2_channel,
                drop=dropout
            )
        else:
            model = GATGraphClassifier(
                num_node_features, num_classes,
                conv1=conv1_channel,
                conv2=conv2_channel,
                heads=gat_heads,
                drop=dropout
            )

        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        best_acc = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer)

            # Evaluate on test set
            te_loss, te_acc, te_prec, te_rec, te_f1 = evaluate_model(model, test_loader)

            # Check for best model
            if te_acc > best_acc:
                best_acc = te_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if enable_early_stopping and epochs_no_improve >= patience:
                print(f"[Fold {fold_index+1}] Early stopping at epoch {epoch+1}")
                break

        # Load best weights
        model.load_state_dict(best_model_wts)

        # Final evaluation for this fold
        final_loss, final_acc, final_prec, final_rec, final_f1 = evaluate_model(model, test_loader)
        print(f"[Fold {fold_index+1}] Test Accuracy = {final_acc:.4f}, F1={final_f1:.4f}")

        test_accuracies.append(final_acc)
        test_precisions.append(final_prec)
        test_recalls.append(final_rec)
        test_f1s.append(final_f1)

        # Save best model
        fold_model_filename = f"best_model_fold_{fold_index+1}.pt"
        torch.save(model.state_dict(), fold_model_filename)

    # Summaries
    mean_acc = np.mean(test_accuracies)
    mean_prec= np.mean(test_precisions)
    mean_rec = np.mean(test_recalls)
    mean_f1  = np.mean(test_f1s)
    std_acc  = np.std(test_accuracies)

    print("=============================================")
    print("Final k-fold Results")
    for i, acc in enumerate(test_accuracies):
        print(f" Fold {i+1}: Acc={acc:.4f}, F1={test_f1s[i]:.4f}")
    print(f" => Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f" => Mean Precision: {mean_prec:.4f}")
    print(f" => Mean Recall: {mean_rec:.4f}")
    print(f" => Mean F1: {mean_f1:.4f}")

    # Save all fold results into CSV
    results_df = pd.DataFrame({
        "Fold":       list(range(1, len(cssgrn_folders)+1)),
        "TestAccuracy":  test_accuracies,
        "TestPrecision": test_precisions,
        "TestRecall":    test_recalls,
        "TestF1":        test_f1s
    })
    results_df.to_csv("kfold_results.csv", index=False)
    print("Saved k-fold CV results (Accuracy, Precision, Recall, F1) to kfold_results.csv")

##########################
## 8. Run the procedure ##
##########################
if __name__ == "__main__":
    run_kfold_experiment()

#########################
## 9. Save hyperparams ##
#########################
variables = [
    "enable_significance_filter",
    "significance_threshold",
    "features_to_remove",
    "model_selected",
    "conv1_channel",
    "conv2_channel",
    "gat_heads",
    "dropout",
    "learning_rate",
    "weight_decay",
    "enable_early_stopping",
    "patience",
    "delta",
    "num_epochs",
    "batch_size"
]

def generate_variable_summary(file_name, variable_names, local_vars):
    summary_lines = []
    for var_name in variable_names:
        value = local_vars.get(var_name, "undefined")
        summary_lines.append(f"{var_name} : {value}")
    summary_content = "\n".join(summary_lines)
    with open(file_name, "w") as file:
        file.write(summary_content)

file_name = "variables_summary.txt"
generate_variable_summary(file_name, variables, locals())



