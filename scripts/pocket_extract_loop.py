import os
from extract_pocket import extract_pocket

PDB_DIR = "data/pdbs"
POCKET_DIR = "data/pockets"

SOLVENTS = ["HOH", "WAT", "NA", "CL", "SO4", "PEG", "MPD", "GOL"]

def find_ligands(pdb_file):
    ligands = set()
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("HETATM"):
                lig = line[17:20].strip()
                if lig not in SOLVENTS:
                    ligands.add(lig)
    return list(ligands)

def main(radius=6.0):
    pdb_files = sorted([f for f in os.listdir(PDB_DIR) if f.endswith(".pdb")])

    for pdb in pdb_files:
        pdb_path = os.path.join(PDB_DIR, pdb)
        ligs = find_ligands(pdb_path)

        if not ligs:
            print(f"{pdb}: No ligand found → skipping (apo structure)")
            continue

        ligand = ligs[0]  # take first ligand
        print(f"{pdb}: extracting pocket for ligand {ligand}")

        try:
            out_file, residues = extract_pocket(pdb_path, ligand, radius)
            print("Pocket saved:", out_file)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main(6.0)
