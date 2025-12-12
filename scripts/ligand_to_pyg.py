#!/usr/bin/env python3
# scripts/ligand_to_pyg.py
"""
Converts data/ligands/*.sdf (or ligand_index.csv SMILES) into PyG atom graphs.
Saves per-ligand torch files into data/pyg/ligands/
"""
import os, glob, csv
from rdkit import Chem
import torch
from torch_geometric.data import Data
import numpy as np

LIG_DIR = "data/ligands"
OUT_DIR = "data/pyg/ligands"
os.makedirs(OUT_DIR, exist_ok=True)

def atom_features_rdkit(atom):
    # atomic num, degree, formal charge, aromatic, num Hs, hybridization
    return [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        1.0 if atom.GetIsAromatic() else 0.0,
        atom.GetTotalNumHs(),
        float(int(atom.GetHybridization()))
    ]

def mol_to_pyg(mol):
    # mol is RDKit Mol with explicit Hs optional
    N = mol.GetNumAtoms()
    feats = [atom_features_rdkit(mol.GetAtomWithIdx(i)) for i in range(N)]
    pos = None
    # Try to get 3D coords if present
    conf = mol.GetConformer() if mol.GetNumConformers()>0 else None
    if conf:
        pos = [list(conf.GetAtomPosition(i)) for i in range(N)]
        pos = torch.tensor(np.array(pos), dtype=torch.float)
    else:
        pos = torch.zeros((N,3), dtype=torch.float)

    x = torch.tensor(np.array(feats), dtype=torch.float)
    # edges via bonds
    edge_index = []
    bond_attr = []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        edge_index.append([i,j]); edge_index.append([j,i])
        bond_attr.append([b.GetBondTypeAsDouble()]); bond_attr.append([b.GetBondTypeAsDouble()])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2,0), dtype=torch.long)
    edge_attr = torch.tensor(np.array(bond_attr), dtype=torch.float) if bond_attr else torch.zeros((0,1), dtype=torch.float)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return data

def main():
    # Option A: use ligand_index.csv to find SMILES/SDF mapping
    csvp = os.path.join(LIG_DIR, "ligand_index.csv")
    if os.path.exists(csvp):
        import pandas as pd
        df = pd.read_csv(csvp)
        for idx,row in df.iterrows():
            smi = row.get("smiles","")
            sdf = row.get("sdf","")
            name = f"{row['pdb']}_{row['ligand_key']}"
            mol = None
            if pd.notna(sdf) and os.path.exists(sdf):
                try:
                    mol = Chem.MolFromMolFile(sdf, removeHs=False)
                except:
                    mol = None
            if mol is None and smi and isinstance(smi,str) and smi.strip():
                try:
                    mol = Chem.MolFromSmiles(smi)
                    mol = Chem.AddHs(mol)
                    Chem.AllChem.EmbedMolecule(mol)  # attempt 3D coords
                except:
                    mol = None
            if mol is None:
                print("Skipping", name, "no mol found")
                continue
            data = mol_to_pyg(mol)
            torch.save(data, os.path.join(OUT_DIR, name + ".pt"))
            print("Wrote ligand:", name)
    else:
        # fallback: convert all sdf files in lig_dir
        sdfs = glob.glob(os.path.join(LIG_DIR,"*.sdf"))
        for s in sdfs:
            base = os.path.splitext(os.path.basename(s))[0]
            mol = Chem.MolFromMolFile(s, removeHs=False)
            if mol is None:
                print("Failed to read", s); continue
            try:
                Chem.AddHs(mol)
                Chem.AllChem.EmbedMolecule(mol)
            except:
                pass
            data = mol_to_pyg(mol)
            torch.save(data, os.path.join(OUT_DIR, base + ".pt"))
            print("Wrote", base)

if __name__ == "__main__":
    main()
