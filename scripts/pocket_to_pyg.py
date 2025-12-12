#!/usr/bin/env python3
# scripts/pocket_to_pyg.py
"""
Converts data/pockets/*.pdb into PyG Data objects.

Produces:
 - data/pyg/pockets_atom/<pdb>_LIGID_atom.pt   (atom-level graphs)
 - data/pyg/pockets_residue/<pdb>_LIGID_res.pt (residue-level graphs)

Residue-level nodes: one node per residue with features:
  - one-hot amino acid (20)
  - residue centroid (pos)
  - residue SASA placeholder (0)  [you can compute real SASA later]

Atom-level nodes: per-atom features:
  - atomic number
  - formal charge
  - aromatic
  - degree
  - hybridization (as int)

Edges: undirected edges between atoms if bond exists (ligand) or if within cutoff (protein)
"""
import os, glob
from Bio import PDB
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem

AA_LIST = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
AA2IDX = {a:i for i,a in enumerate(AA_LIST)}

PDB_DIR = "data/pockets"
OUT_ATOM = "data/pyg/pockets_atom"
OUT_RES = "data/pyg/pockets_residue"

os.makedirs(OUT_ATOM, exist_ok=True)
os.makedirs(OUT_RES, exist_ok=True)

def atom_features_biopython(atom):
    # atom is Bio.PDB.Atom.Atom
    an = atom.element if atom.element else atom.get_name()[0]
    try:
        at_num = Chem.GetPeriodicTable().GetAtomicNumber(an.capitalize())
    except Exception:
        # fallback: simple mapping
        try:
            at_num = int(''.join([c for c in an if c.isdigit()]))
        except:
            at_num = 0
    # basic features: atomic number, formal charge (unknown here), is_aromatic (0)
    return [float(at_num), 0.0, 0.0, float(atom.get_bfactor()) if atom.get_bfactor() else 0.0]

def residue_centroid(res):
    coords = [a.get_coord() for a in res.get_atoms()]
    return np.mean(coords, axis=0)

def build_residue_graph(structure):
    residues = [r for r in structure.get_residues() if r.get_id()[0] == ' ']
    N = len(residues)
    node_feats = []
    pos = []
    for r in residues:
        resname = r.get_resname()
        onehot = np.zeros(len(AA_LIST), dtype=float)
        idx = AA2IDX.get(resname, None)
        if idx is not None:
            onehot[idx] = 1.0
        centroid = residue_centroid(r)
        feats = np.concatenate([onehot, np.array([0.0])])  # placeholder extra scalar
        node_feats.append(feats)
        pos.append(centroid)
    node_feats = torch.tensor(np.vstack(node_feats), dtype=torch.float) if node_feats else torch.zeros((0,len(AA_LIST)+1))
    pos = torch.tensor(np.vstack(pos), dtype=torch.float) if pos else torch.zeros((0,3))

    # Build edges: connect residues if centroid distance < 8.0 A
    edge_index = []
    for i in range(N):
        for j in range(i+1,N):
            d = np.linalg.norm(pos[i].numpy() - pos[j].numpy())
            if d <= 8.0:
                edge_index.append([i,j])
                edge_index.append([j,i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2,0),dtype=torch.long)

    data = Data(x=node_feats, pos=pos, edge_index=edge_index)
    return data

def build_atom_graph_from_pdb(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    st = parser.get_structure('X', pdb_path)
    atoms = [a for a in st.get_atoms()]
    N = len(atoms)
    feats = []
    pos = []
    for a in atoms:
        feats.append(atom_features_biopython(a))
        pos.append(a.get_coord())
    feats = torch.tensor(np.array(feats), dtype=torch.float) if feats else torch.zeros((0,4))
    pos = torch.tensor(np.array(pos), dtype=torch.float) if pos else torch.zeros((0,3))

    # Build edges: connect atoms if distance <= 1.9 A (bond-like) or <= 4.5 A (nonbond) as a fallback
    edge_index = []
    coords = np.array(pos)
    for i in range(N):
        for j in range(i+1,N):
            d = np.linalg.norm(coords[i] - coords[j])
            if d <= 1.9:
                edge_index.append([i,j])
                edge_index.append([j,i])
            # optionally include longer-range edges (commented)
            # elif d <= 4.5:
            #    edge_index.append([i,j]); edge_index.append([j,i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2,0),dtype=torch.long)
    data = Data(x=feats, pos=pos, edge_index=edge_index)
    return data

def main():
    pdbs = sorted(glob.glob(os.path.join(PDB_DIR,"*.pdb")))
    for p in pdbs:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            atom_data = build_atom_graph_from_pdb(p)
            res_data = build_residue_graph(PDB.PDBParser(QUIET=True).get_structure('X', p))
            torch.save(atom_data, os.path.join(OUT_ATOM, base + "_atom.pt"))
            torch.save(res_data, os.path.join(OUT_RES, base + "_res.pt"))
            print("Saved:", base)
        except Exception as e:
            print("ERROR processing", p, e)

if __name__ == "__main__":
    main()

