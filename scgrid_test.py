import os
import glob
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

######################
## 1. Global config ##
######################
WORKSPACE_PATH = "" # <path_to_workspace>
if not os.path.exists(WORKSPACE_PATH):
    os.makedirs(WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

# --------------------- Model & Feature Params --------------------- #
model_type     = "GAT"  # "GCN" or "GAT"
conv1_channel  = 64
conv2_channel  = 128
gat_heads      = 4
dropout        = 0.1

# -------------------- Normalized Entropy Threshold ---------------- #
# If normalized_entropy > ENTROPY_THRESHOLD, label = "Unknown".
# Typically pick between 0.0 and 1.0:
#   0.0 means "reject if model is not perfectly certain" 
#       (practically everything becomes "Unknown")
#   1.0 means "never reject"
ENTROPY_THRESHOLD = 0.9

# -------------------------- Feature Ablation ---------------------- #
# 0 = avg_expr, 1 = fold_change, 2 = pct1, 3 = pct2
features_to_remove = None # [0] to remove avg_expr for example, default = None
raw_num_node_features = 4   # 4 features by default

num_removed = len(features_to_remove) if features_to_remove is not None else 0
num_node_features = max(raw_num_node_features - num_removed, 1)



# ------------------------ Trained Model Path ----------------------- #
MODEL_PATH = "" # <path_to_model>

# -------------------- CSSGRN & Node Features ----------------------- #
test_input_path = "" # <path_to_cssgrn_folder>
avg_test_path   = "" # <path_to_avg_node_feature>
fold_change_test_path = "" # <path_to_fc_node_feature>
pct1_test_path  = "" # <path_to_pct1_node_feature>
pct2_test_path  = "" # <path_to_pct2_node_feature>

OUTPUT_CSV      = "" # <name_of_output_csv>

# ----------------------- Cell Types / Labels ------------------------ #
output_classes = [
    'B cell', 'erythrocyte', 'Fraction A pre-pro B cell',
    'granulocyte', 'hematopoietic stem cell', 'macrophage',
    'monocyte', 'T cell'
]

graph_labels = {cell_type: i for i, cell_type in enumerate(output_classes)}

enable_significance_filter = False
significance_threshold = 1e-5

#########################
## 2. Helper functions ##
#########################
def load_gene_features(avg_path, fold_path, pct1_path, pct2_path):
    avg_expr_df      = pd.read_csv(avg_path,      index_col=0)
    fold_change_df   = pd.read_csv(fold_path,     index_col=0)
    pct1_df          = pd.read_csv(pct1_path,     index_col=0)
    pct2_df          = pd.read_csv(pct2_path,     index_col=0)
    return avg_expr_df, fold_change_df, pct1_df, pct2_df

def compute_normalized_entropy(prob_vector):
    """
    prob_vector: 1D iterable of probabilities for each class.
    Returns the normalized entropy: H(p)/log(K).
    """
    EPS = 1e-9
    K   = len(prob_vector)
    H   = 0.0
    for p in prob_vector:
        if p > 0:
            H -= p * math.log(p + EPS)
    
    H_max = math.log(K + EPS)
    return H / H_max

def ablate_node_features(data, features_to_remove):
    """
    Given a single PyG Data object whose data.x is
    (num_nodes × raw_num_node_features),
    remove the requested feature‐columns.
    """
    # handle single int
    if isinstance(features_to_remove, int):
        features_to_remove = [features_to_remove]
    if features_to_remove is None or (isinstance(features_to_remove, list) and len(features_to_remove) == 0):
        return data

    # columns to keep
    all_idxs = list(range(data.x.size(1)))
    keep_idxs = [i for i in all_idxs if i not in features_to_remove]

    if len(keep_idxs) == 0:
        # if you drop everything, fallback to a single constant feature
        data.x = torch.ones((data.num_nodes, 1))
    else:
        data.x = data.x[:, keep_idxs]

    return data

####################################
## 3. Model definitions (GCN/GAT) ##
####################################
class GCNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, conv1=16, conv2=32, drop=0.0):
        super(GCNGraphClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, conv1)
        self.conv2 = GCNConv(conv1, conv2)
        self.fc    = torch.nn.Linear(conv2, num_classes)
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
        self.fc    = torch.nn.Linear(conv2, num_classes)
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

############################################
## 4. Build graph from adjacency csv file ##
############################################
def build_graph(csv_path, cell_type, 
                avg_expr_df, fold_change_df, pct1_df, pct2_df,
                enable_significance_filter=False, sig_threshold=1e-5):
    """
    Reads adjacency CSV for the given cell_type, builds a single PyG Data object 
    with node features from the query DataFrames.
    """
    df = pd.read_csv(csv_path)
    
    if enable_significance_filter:
        df = df[df['p'] < sig_threshold].copy()
    
    unique_genes = pd.concat([df['source'], df['target']]).unique()
    gene_to_int  = {g: i for i, g in enumerate(unique_genes)}
    int_to_gene  = {i: g for g, i in gene_to_int.items()}

    node_features = []
    for gene in unique_genes:
        if (gene in avg_expr_df.index and
            gene in fold_change_df.index and
            gene in pct1_df.index and
            gene in pct2_df.index):
            
            avg_val  = avg_expr_df.loc[gene, cell_type]
            fold_val = fold_change_df.loc[gene, cell_type]
            pct1_val = pct1_df.loc[gene, cell_type]
            pct2_val = pct2_df.loc[gene, cell_type]
            node_features.append([avg_val, fold_val, pct1_val, pct2_val])
        else:
            node_features.append([0.0, 0.0, 0.0, 0.0])
    
    df['source'] = df['source'].map(gene_to_int)
    df['target'] = df['target'].map(gene_to_int)
    
    edge_index = torch.tensor([df['source'].values, df['target'].values], dtype=torch.long)
    edge_attr  = torch.tensor(df['coef_mean'].values, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )
    data.node_names = [int_to_gene[i] for i in range(x.size(0))]
    return data

####################################
## 5. Save classification results ##
####################################
def save_classification_results(
    file_names,
    predicted_labels,
    probabilities,
    entropies,
    label_to_classname,
    output_csv
):
    """
    file_names: adjacency file names
    predicted_labels: either integer indices or string "Unknown"
    probabilities: 2D list, row = probability distribution over classes
    label_to_classname: e.g. {0: "B cell", 1: "erythrocyte", ...}
    output_csv: output CSV file path
    """
    stripped_file_names = [name.replace("CSSGRN_", "").replace(".csv", "") for name in file_names]
    
    # Convert predicted_labels to a final string
    final_preds = []
    for label in predicted_labels:
        if isinstance(label, str) and label == "Unknown":
            final_preds.append("Unknown")
        else:
            final_preds.append(label_to_classname.get(label, "Unknown"))
    
    df = pd.DataFrame({
        "CSSGRN_File": stripped_file_names,
        "Predicted_Class": final_preds
    })
    
    prob_df = pd.DataFrame(probabilities)
  
    # Create column headers
    class_prob_cols = []
    for i in range(len(label_to_classname)):
        class_name = label_to_classname[i]
        class_prob_cols.append(f"Prob_{class_name}")
    prob_df.columns = class_prob_cols
    
    final_df = pd.concat([df, prob_df], axis=1)
    final_df["Entropy"] = entropies
  
    # Reorder rows to match the original output_classes order
    final_df = final_df.sort_values("CSSGRN_File").reset_index(drop=True)
    
    final_df.to_csv(output_csv, index=False)
    print(f"[Info] Classification results (with NE-based rejection) written to {output_csv}")

###################
## 6. Main logic ##
###################
def classify_dataset():
    # 1) Load the 4 node‑feature tables as before
    avg_expr_df, fold_change_df, pct1_df, pct2_df = load_gene_features(
        avg_test_path,
        fold_change_test_path,
        pct1_test_path,
        pct2_test_path
    )

    # 2) Initialize and load your model
    num_classes       = len(output_classes)
    if model_type == "GCN":
        model = GCNGraphClassifier(num_node_features, num_classes,
                                   conv1=conv1_channel,
                                   conv2=conv2_channel,
                                   drop=dropout)
    else:
        model = GATGraphClassifier(num_node_features, num_classes,
                                   conv1=conv1_channel,
                                   conv2=conv2_channel,
                                   heads=gat_heads,
                                   drop=dropout)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Reverse map for predictions
    reverse_label_map = {v:k for k,v in graph_labels.items()}

    # 3) Discover all query CSVs and extract “Cluster0”, “Cluster1”, …
    query_paths = sorted(glob.glob(os.path.join(test_input_path, "CSSGRN_*.csv")))
    query_names = [os.path.splitext(os.path.basename(p))[0].replace("CSSGRN_","")
                   for p in query_paths]

    file_names       = []   # will store Cluster0, Cluster1, …
    predicted_labels = []   # will store indices or "Unknown"
    probabilities    = []   # will store per‑class probability vectors
    entropies        = []   # will store entropy values

    with torch.no_grad():
        for q_name, csv_path in zip(query_names, query_paths):
            data = build_graph(
                csv_path=csv_path,
                cell_type=q_name,
                avg_expr_df=avg_expr_df,
                fold_change_df=fold_change_df,
                pct1_df=pct1_df,
                pct2_df=pct2_df,
                enable_significance_filter=enable_significance_filter,
                sig_threshold=significance_threshold
            )
            
            data = ablate_node_features(data, features_to_remove)
            
            # single‑graph batch
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)

            # forward pass
            output  = model(data)            # [1, num_classes]
            probs   = torch.exp(output)[0]   # [num_classes]
            probs_np = probs.cpu().numpy()

            # decide reject vs accept
            ne = compute_normalized_entropy(probs_np)
            entropies.append(ne)
            if ne > ENTROPY_THRESHOLD:
                pred_lbl = "Unknown"
            else:
                pred_lbl = output.argmax(dim=1).item()

            file_names.append(q_name)
            predicted_labels.append(pred_lbl)
            probabilities.append(probs_np.tolist())

    # 4) Save results exactly as before
    save_classification_results(
        file_names=file_names,
        predicted_labels=predicted_labels,
        probabilities=probabilities,
        entropies=entropies,
        label_to_classname=reverse_label_map,
        output_csv=OUTPUT_CSV
    )

if __name__ == "__main__":
    classify_dataset()

########################
## 7. Save parameters ##
########################
variables = ["model_type", "conv1_channel", "conv2_channel", "gat_heads", "dropout", "ENTROPY_THRESHOLD"]
def generate_variable_summary(file_name, variable_names, local_vars):
    summary_lines = []
    for var_name in variable_names:
        value = local_vars.get(var_name, "undefined")
        summary_lines.append(f"{var_name} : {value}")
    
    with open(file_name, "w") as file:
        file.write("\n".join(summary_lines))

file_name = "variables_summary.txt"
generate_variable_summary(file_name, variables, locals())

