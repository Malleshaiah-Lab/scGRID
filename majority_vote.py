# src/scgrid/majority_vote.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
from glob import glob


def majority_vote(input_dir: Path, out_csv: Path) -> Path:
    input_dir = Path(input_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = glob(str(input_dir / "*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found in {input_dir!r}")

    df0 = pd.read_csv(csv_paths[0])
    classes = [c.replace("Prob_", "") for c in df0.columns if c.startswith("Prob_")]
    if "Unknown" not in classes:
        classes.append("Unknown")

    votes = {}
    for path in csv_paths:
        df = pd.read_csv(path, usecols=["CTSGRN_File", "Predicted_Class"])
        for ctsfile, pred in zip(df["CTSGRN_File"], df["Predicted_Class"]):
            votes.setdefault(ctsfile, dict.fromkeys(classes, 0))
            votes[ctsfile][pred] += 1

    rows = []
    for ctsfile, cnts in votes.items():
        max_votes = max(cnts.values())
        winners = [cls for cls, v in cnts.items() if v == max_votes]
        majority = winners[0] if len(winners) == 1 else "Unknown"

        row = {"CTSGRN_File": ctsfile, "Predicted_Class": majority}
        for cls in classes:
            row[f"Vote_{cls}"] = cnts.get(cls, 0)
        rows.append(row)

    pd.DataFrame(rows).sort_values("CTSGRN_File").to_csv(out_csv, index=False)
    return out_csv
