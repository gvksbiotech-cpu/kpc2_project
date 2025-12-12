#!/usr/bin/env python3
# scripts/build_dataset.py
"""
Pairs pocket PyG objects with ligand PyG objects using ligand_index.csv and pdb_metadata.csv.
Outputs:
 - data/pyg/dataset/  (each entry saved as pair_<i>.pt containing dict {'pocket':pocket_data, 'ligand':lig_data, 'meta':meta})
 - data/pyg/dataset_index.csv  (mapping rows)
"""
import os, csv, glob, torch
import pandas as pd

POCKET_PYG_ATOM_DIR = "data/pyg/pockets_atom"
LIG_PYG_DIR = "data/pyg/ligands"
OUT_DIR = "data/pyg/dataset"
os.makedirs(OUT_DIR, exist_ok=True)

lig_csv = "data/ligands/ligand_index.csv"
meta_csv = "data/pdb_metadata.csv"

if not os.path.exists(lig_csv):
    print("ligand_index.csv not found. Run extract_ligands.py first.")
    raise SystemExit

df = pd.read_csv(lig_csv)
meta = pd.read_csv(meta_csv) if os.path.exists(meta_csv) else None

rows = []
counter = 0
for idx, row in df.iterrows():
    pdb = row['pdb']
    ligand_key = row['ligand_key']
    pocket_basename = None
    # find matching pocket atom file: pocket files have pattern <pdb>_<ligand>_pocket.pdb saved as <pdb>_<ligand>_pocket_atom.pt when processed
    # simpler: search pockets_atom for files starting with pdb
    candidates = [f for f in os.listdir(POCKET_PYG_ATOM_DIR) if f.startswith(pdb)]
    if not candidates:
        # no pocket resource for this pdb
        continue
    # attempt to match ligand_key token inside candidate name
    matched = None
    for c in candidates:
        if ligand_key.split("_")[0] in c:
            matched = c; break
    if matched is None:
        matched = candidates[0]
    pocket_path = os.path.join(POCKET_PYG_ATOM_DIR, matched)
    ligand_name = f"{row['pdb']}_{ligand_key}.pt"
    ligand_path = os.path.join(LIG_PYG_DIR, ligand_name)
    if not os.path.exists(pocket_path) or not os.path.exists(ligand_path):
        # skip if any missing object
        continue
    pocket_data = torch.load(pocket_path)
    ligand_data = torch.load(ligand_path)
    meta_entry = {'pdb':pdb, 'ligand_key':ligand_key, 'smiles':row.get('smiles','')}
    outpath = os.path.join(OUT_DIR, f"pair_{counter}.pt")
    torch.save({'pocket':pocket_data, 'ligand':ligand_data, 'meta':meta_entry}, outpath)
    rows.append({'pair_index':counter, 'pdb':pdb, 'ligand_key':ligand_key, 'pocket_file':pocket_path, 'ligand_file':ligand_path, 'out_file':outpath})
    counter += 1

# write index CSV
out_idx = "data/pyg/dataset_index.csv"
pd.DataFrame(rows).to_csv(out_idx, index=False)
print("Wrote dataset pairs:", len(rows), "index at", out_idx)
