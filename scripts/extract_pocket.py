from Bio import PDB
import numpy as np
import os, sys

def extract_pocket(pdb_path, ligand_resname, radius=6.0, out_dir="data/pockets"):
    parser = PDB.PDBParser(QUIET=True)
    st = parser.get_structure('X', pdb_path)
    
    ligand_atoms = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() == ligand_resname:
                    for a in res:
                        ligand_atoms.append(a.get_coord())

    if len(ligand_atoms) == 0:
        raise ValueError(f"Ligand '{ligand_resname}' not found in {pdb_path}")

    ligand_coords = np.array(ligand_atoms)
    pocket_residues = []

    for model in st:
        for chain in model:
            for res in chain:
                for a in res:
                    dist = np.linalg.norm(ligand_coords - a.get_coord(), axis=1)
                    if np.min(dist) <= radius:
                        pocket_residues.append(res)
                        break

    os.makedirs(out_dir, exist_ok=True)
    out_pdb = os.path.join(out_dir, os.path.basename(pdb_path).replace(".pdb","") 
                           + f"_{ligand_resname}_pocket.pdb")

    io = PDB.PDBIO()

    class PocketSelect(PDB.Select):
        def __init__(self, residues):
            self.residues = residues

        def accept_residue(self, residue):
            return residue in self.residues

    io.set_structure(st)
    io.save(out_pdb, PocketSelect(pocket_residues))
    
    return out_pdb, pocket_residues


if __name__ == "__main__":
    pdb_file = sys.argv[1]
    ligand = sys.argv[2]
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0

    out_file, residues = extract_pocket(pdb_file, ligand, radius)
    print("Pocket saved to:", out_file)
    print("Residues:")
    for r in residues:
        print(r.get_resname(), r.get_id()[1])
