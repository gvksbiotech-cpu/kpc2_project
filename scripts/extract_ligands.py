#!/usr/bin/env python3
"""
scripts/extract_ligands.py

- Scans data/pdbs/*.pdb for non-solvent HETATM residues.
- Writes a ligand PDB fragment for each ligand, converts to SDF using RDKit (fallback to OpenBabel if needed).
- Writes a CSV index: data/ligands/ligand_index.csv with columns: pdb, ligand_key, resname, sdf, smiles
"""
import os, csv, subprocess
from rdkit import Chem

PDB_DIR = "data/pdbs"
LIG_DIR = "data/ligands"
OUT_CSV = os.path.join(LIG_DIR, "ligand_index.csv")
SOLVENTS = set(["HOH","WAT","NA","CL","SO4","GOL","MPD","PEG","EDO","DMS"])

os.makedirs(LIG_DIR, exist_ok=True)

def extract_het_atoms(pdb_file):
    lig_dict = {}
    with open(pdb_file) as fh:
        for line in fh:
            if line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname in SOLVENTS or not resname:
                    continue
                resseq = line[22:26].strip()
                chain = line[21].strip() or "_"
                key = f"{resname}_{chain}{resseq}"
                lig_dict.setdefault(key, []).append(line)
    return lig_dict

def write_temp_pdb(atom_lines, out_tmp):
    with open(out_tmp, "w") as fh:
        fh.writelines(atom_lines)

def rdkit_from_pdb(tmp_pdb, out_sdf):
    # RDKit can read ligand-only PDB fragments but may fail to infer bonds reliably.
    try:
        mol = Chem.MolFromPDBFile(tmp_pdb, removeHs=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol, catchErrors=True)
        smi = Chem.MolToSmiles(mol)
        w = Chem.SDWriter(out_sdf)
        w.write(mol)
        w.close()
        return smi
    except Exception:
        return None

def obabel_convert(tmp_pdb, out_sdf):
    # fallback using openbabel CLI (obabel)
    cmd = ["obabel", tmp_pdb, "-O", out_sdf, "--gen3d"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # try reading with rdkit
        m = Chem.MolFromMolFile(out_sdf, removeHs=False)
        if m:
            return Chem.MolToSmiles(m)
        return None
    except Exception:
        return None

def main():
    rows = []
    pdbs = sorted([f for f in os.listdir(PDB_DIR) if f.lower().endswith(".pdb")])
    for p in pdbs:
        path = os.path.join(PDB_DIR, p)
        ligs = extract_het_atoms(path)
        if not ligs:
            print(f"{p}: no non-solvent HETATM ligands found.")
            continue
        for key, lines in ligs.items():
            tmp = os.path.join(LIG_DIR, f"tmp_{p}_{key}.pdb")
            out_sdf = os.path.join(LIG_DIR, f"{os.path.splitext(p)[0]}_{key}.sdf")
            write_temp_pdb(lines, tmp)
            smiles = rdkit_from_pdb(tmp, out_sdf)
            if not smiles:
                smiles = obabel_convert(tmp, out_sdf)
            if not smiles:
                print("Failed to convert ligand", key, "from", p)
                # still write a placeholder
                rows.append({"pdb":p,"ligand_key":key,"resname":key.split("_")[0],"sdf":out_sdf,"smiles":""})
            else:
                rows.append({"pdb":p,"ligand_key":key,"resname":key.split("_")[0],"sdf":out_sdf,"smiles":smiles})
                print("Wrote", out_sdf, "SMILES:", smiles)
            # clean tmp
            try:
                os.remove(tmp)
            except:
                pass

    # write CSV
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb","ligand_key","resname","sdf","smiles"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Wrote ligand index to", OUT_CSV)

if __name__ == "__main__":
    main()

