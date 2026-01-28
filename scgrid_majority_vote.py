import os
import pandas as pd
from glob import glob

WORKSPACE_PATH = "" # <path_to_workspace>
if not os.path.exists(WORKSPACE_PATH):
    os.makedirs(WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

# ────────── CONFIG ──────────
INPUT_DIR = "" # <path_to_gat_models>
OUTPUT_CSV = os.path.join(WORKSPACE_PATH, "majority_vote_results.csv")
# ───────────────────────────

# 1) find all CSVs
csv_paths = glob(os.path.join(INPUT_DIR, "*.csv"))
if not csv_paths:
    raise FileNotFoundError(f"No CSVs found in {INPUT_DIR!r}")

# 2) infer the full set of possible classes from the 'Prob_*' columns of the first file
df0 = pd.read_csv(csv_paths[0])
classes = [col.replace("Prob_", "") for col in df0.columns if col.startswith("Prob_")]

if "Unknown" not in classes:
    classes.append("Unknown")

# 3) tally votes
votes = {}
for path in csv_paths:
    df = pd.read_csv(path, usecols=["CSSGRN_File", "Predicted_Class"])
    for cssfile, pred in zip(df["CSSGRN_File"], df["Predicted_Class"]):
        if cssfile not in votes:
            votes[cssfile] = dict.fromkeys(classes, 0)
        votes[cssfile][pred] += 1

# 4) build the result DataFrame
rows = []
for cssfile, cnts in votes.items():
    max_votes = max(cnts.values()) # pick the class with the highest vote
    winners = [cls for cls, v in cnts.items() if v == max_votes]
    if len(winners) == 1:
        majority = winners[0]
    else:
        majority = "Unknown"
    row = {
        "CSSGRN_File": cssfile,
        "Predicted_Class": majority,
    }
    # add vote counts for each class
    for cls in classes:
        row[f"Vote_{cls}"] = cnts.get(cls, 0)
    rows.append(row)

result = pd.DataFrame(rows).sort_values("CSSGRN_File")

# 5) save
result.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote majority‐vote table to:\n  {OUTPUT_CSV}")
