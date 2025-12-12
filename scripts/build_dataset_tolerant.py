#!/usr/bin/env python3
import os, glob, torch, pandas as pd
import torch_geometric
from torch_geometric.data import Data as _TorchGeoData
torch.serialization.add_safe_globals([_TorchGeoData])

POCKET_DIR = "data/pyg/pockets_atom"
LIG_DIR = "data/pyg/ligands"
OUT_DIR = "data/pyg/dataset"
os.makedirs(OUT_DIR, exist_ok=True)

# read ligand index mapping (optional)
lig_csv = "data/ligands/ligand_index.csv"
df = pd.read_csv(lig_csv) if os.path.exists(lig_csv) else None

pocket_files = sorted(glob.glob(os.path.join(POCKET_DIR,"*.pt")))
lig_files = sorted(glob.glob(os.path.join(LIG_DIR,"*.pt")))

def canonical_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def find_lig_for_pdb(pdb):
    # Highest priority: ligand file that starts with pdb_
    for lf in lig_files:
        if os.path.basename(lf).startswith(pdb + "_"):
            return lf
    # Next: any ligand file that starts with pdb
    for lf in lig_files:
        if os.path.basename(lf).startswith(pdb):
            return lf
    # Next: if ligand_index.csv available, try constructing name from it
    if df is not None:
        rows = df[df['pdb']==pdb]
        for _,r in rows.iterrows():
            candidate = os.path.join(LIG_DIR, f"{r['pdb']}_{r['ligand_key']}.pt")
            if os.path.exists(candidate):
                return candidate
    # Next: try any ligand file that contains the pdb id anywhere
    for lf in lig_files:
        if pdb in os.path.basename(lf):
            return lf
    # give up
    return None

rows=[]
count=0
for pf in pocket_files:
    pdb = os.path.basename(pf).split("_")[0]
    lf = find_lig_for_pdb(pdb)
    if lf is None:
        print("No ligand .pt for pocket:", pf)
        continue
    pocket_data = torch.load(pf, weights_only=False)
    ligand_data = torch.load(lf, weights_only=False)
    out = os.path.join(OUT_DIR, f"pair_{count}.pt")
    torch.save({'pocket':pocket_data, 'ligand':ligand_data, 'meta':{'pdb':pdb, 'ligand_file':os.path.basename(lf)}}, out)
    rows.append({'pair_index':count, 'pocket':pf, 'ligand':lf, 'out':out})
    count+=1

pd.DataFrame(rows).to_csv("data/pyg/dataset_index.csv", index=False)
print("Wrote", count, "pairs, index saved to data/pyg/dataset_index.csv")
