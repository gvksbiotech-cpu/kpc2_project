#!/usr/bin/env python3
import os, glob, csv
PA = "data/pyg/pockets_atom"
PL = "data/pyg/ligands"
LCSV = "data/ligands/ligand_index.csv"

pockets = sorted(os.listdir(PA)) if os.path.isdir(PA) else []
ligands = sorted(os.listdir(PL)) if os.path.isdir(PL) else []
print("pockets_atom count:", len(pockets))
print("ligands pyg count:", len(ligands))
print("\nfirst 20 pockets:")
for p in pockets[:20]:
    print("  ", p)
print("\nfirst 20 ligands:")
for l in ligands[:20]:
    print("  ", l)

rows = []
if os.path.exists(LCSV):
    with open(LCSV) as fh:
        r = csv.DictReader(fh)
        for i,row in enumerate(r):
            rows.append((row['pdb'], row['ligand_key']))
print("\nfirst 20 ligand_index.csv rows:")
for r in rows[:20]:
    print("  ", r)

print("\nChecking candidates (for first 30 ligand_index entries):")
missing=[]
for pdb, lig in rows[:30]:
    # candidate pockets starting with pdb
    cand_pocks = [p for p in pockets if p.startswith(pdb)]
    # candidate ligand pyg files that start with pdb or contain the ligand resname
    cand_ligs = [l for l in ligands if l.startswith(pdb) or lig.split('_')[0] in l]
    print(f"{pdb} {lig} -> pocket candidates: {cand_pocks[:5]} ; ligand candidates: {cand_ligs[:5]}")
    if not cand_pocks or not cand_ligs:
        missing.append((pdb, lig, cand_pocks, cand_ligs))
print("\nMISSING (sample up to 50):")
for m in missing[:50]:
    print(" ", m)
print("\nTotal missing pairs:", len(missing))
