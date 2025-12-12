#!/usr/bin/env python3
"""
Label dataset pairs using InChIKey matching (robust against SMILES variations).
"""
import os
import pandas as pd
from rdkit import Chem

def smi_to_inchikey(s):
    """Convert SMILES → InChIKey safely."""
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.InchiToInchiKey(Chem.MolToInchi(m))
    except:
        return None

def normalize_smiles(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.lower() in ("nan", "none", ""):
        return None
    return s

def build_activity_map(df):
    """
    Returns: dict {inchikey → activity_label}
    Priority: pchembl_value > computed pIC50 from standard_value/units
    """
    act_map = {}

    for _, row in df.iterrows():
        smi = normalize_smiles(row.get("canonical_smiles"))
        if smi is None:
            continue

        ik = smi_to_inchikey(smi)
        if ik is None:
            continue

        label = None

        # Prefer pchembl_value
        pch = row.get("pchembl_value")
        try:
            if pch is not None and str(pch).lower() not in ("nan","none",""):
                label = float(pch)
        except:
            pass

        # Try converting standard_value + units
        if label is None:
            val = row.get("standard_value")
            units = row.get("standard_units")
            try:
                v = float(val)
                if v > 0 and isinstance(units, str):
                    u = units.lower()
                    import math
                    if "nm" in u:
                        label = 9 - math.log10(v)
                    elif "um" in u or "µm" in u:
                        label = 6 - math.log10(v)
                    elif "mm" in u:
                        label = 3 - math.log10(v)
            except:
                pass

        if label is None:
            continue

        # If duplicate keys exist, keep highest activity
        if ik not in act_map or label > act_map[ik]:
            act_map[ik] = label

    return act_map

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", required=True)
    parser.add_argument("--ligand_index", required=True)
    args = parser.parse_args()

    act_df = pd.read_csv(args.activity, dtype=str)
    lig_df = pd.read_csv(args.ligand_index, dtype=str)

    act_map = build_activity_map(act_df)
    print("Activity InChIKeys:", len(act_map))

    rows = []
    for _, row in lig_df.iterrows():
        pdb = row.get("pdb")
        ligfile = row.get("ligand_file")
        smi = normalize_smiles(row.get("canonical_smiles") or row.get("smiles"))
        ik = smi_to_inchikey(smi) if smi else None

        label = act_map.get(ik, "")
        note = "ok" if ik in act_map else "missing"

        rows.append({
            "pdb": pdb,
            "ligand_file": ligfile,
            "smiles": smi,
            "inchikey": ik,
            "label": label,
            "note": note
        })

    out = pd.DataFrame(rows)
    os.makedirs("data/pyg", exist_ok=True)
    out.to_csv("data/pyg/dataset_labeling_summary_inchikey.csv", index=False)

    print("Written to data/pyg/dataset_labeling_summary_inchikey.csv")
    print(out["note"].value_counts())

if __name__ == "__main__":
    main()
