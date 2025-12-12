#!/usr/bin/env python3
"""
scripts/add_labels.py

Annotate data/pyg/dataset/pair_*.pt with meta['y'] (numeric activity label, pIC50).
Usage:
    python scripts/add_labels.py  # uses defaults
Options:
    --activity PATH    path to activity CSV (default data/activity/chembl_activity.csv)
    --ligand_index PATH path to ligand_index.csv (default data/ligands/ligand_index.csv)
    --force            overwrite existing meta['y'] values
"""
import os, glob, argparse, math
import pandas as pd
import torch

def safe_float(x):
    try:
        return float(x)
    except:
        return None


def compute_pic50_from_standard(val, units):
    """
    Safe conversion of standard_value + units → pIC50.
    Handles NaN, missing units, bad strings.
    Returns None when conversion is not possible.
    """
    import math
    import pandas as _pd

    # Convert value → float
    try:
        v = float(val)
    except Exception:
        return None

    if v <= 0:
        return None

    # Units clean-up
    if units is None:
        return None
    if isinstance(units, float) and (_pd.isna(units)):
        return None

    u = str(units).strip().lower()

    # Unit cases
    if "nm" in u:
        return 9 - math.log10(v)
    if "um" in u or "µm" in u:
        return 6 - math.log10(v)
    if "mm" in u:
        return 3 - math.log10(v)
    if u in ("m", "mol", "molar"):
        return -math.log10(v)

    # Unknown units → cannot convert safely
    return None
    if val is None:
        return None
    v = safe_float(val)
    if v is None:
        return None
    if units is None:
        return None
    u = units.strip().lower()
    if 'nm' in u:
        # val in nM -> pIC50 = 9 - log10(val_nM)
        if v <= 0:
            return None
        return 9.0 - math.log10(v)
    if 'um' in u or 'µm' in u or 'micromolar' in u:
        if v <= 0:
            return None
        return 6.0 - math.log10(v)
    if 'mm' in u:
        if v <= 0:
            return None
        return 3.0 - math.log10(v)
    if 'm' == u or u == 'mol/l' or 'mol' in u:
        # value in M
        if v <= 0:
            return None
        return -math.log10(v)
    # fallback: assume nM if unit unclear (conservative)
    if v > 1e-3:
        # assume nM
        return 9.0 - math.log10(v)
    return None

def normalize_smi(s):
    if pd.isna(s):
        return None
    return str(s).strip().lower()

def build_activity_map(act_df):
    """
    Build mapping SMILES -> pIC50 (float)
    Preferences:
      - use 'pchembl_value' if present
      - else use standard_value + standard_units (convert to pIC50)
    For duplicates, take median pIC50.
    """
    smi_to_vals = {}
    if 'pchembl_value' in act_df.columns:
        for _,r in act_df.dropna(subset=['pchembl_value']).iterrows():
            smi = normalize_smi(r.get('canonical_smiles') or r.get('smiles') or r.get('SMILES'))
            if not smi:
                continue
            v = safe_float(r['pchembl_value'])
            if v is None:
                continue
            smi_to_vals.setdefault(smi, []).append(v)
    # fallback to standard_value
    for _,r in act_df.iterrows():
        smi = normalize_smi(r.get('canonical_smiles') or r.get('smiles') or r.get('SMILES'))
        if not smi:
            continue
        if 'pchembl_value' in r and not pd.isna(r['pchembl_value']):
            continue
        std = r.get('standard_value', None)
        units = r.get('standard_units', r.get('units', None))
        pic50 = compute_pic50_from_standard(std, units)
        if pic50 is not None:
            smi_to_vals.setdefault(smi, []).append(pic50)
    # collapse to median
    smi_map = {}
    for k,v in smi_to_vals.items():
        try:
            smi_map[k] = float(pd.Series(v).median())
        except:
            continue
    return smi_map

def main(args):
    ligand_index = args.ligand_index
    activity_csv = args.activity
    force = args.force

    if not os.path.exists(ligand_index):
        print("Ligand index not found:", ligand_index)
        return
    ligdf = pd.read_csv(ligand_index)
    # build mapping: key -> smiles
    # key used in pair metadata may be file basename like "<pdb>_<ligkey>.pt" or similar.
    # We'll map by pdb + ligand_key and also keep SMILES->pdbKey
    lig_map = {}
    for _,r in ligdf.iterrows():
        pdb = r.get('pdb')
        key = r.get('ligand_key')
        smi = normalize_smi(r.get('smiles', None))
        if pd.isna(key):
            continue
        # create filenames that might have been used in pair meta 'ligand_file' (various styles)
        # common forms: <pdb>_<key>.pt  OR <pdb>.pdb_<key>.pt  OR <pdb>_<key>.sdf (but pair files use .pt)
        possible_names = set()
        if pdb and key:
            possible_names.add(f"{pdb}_{key}.pt")
            possible_names.add(f"{pdb}.pdb_{key}.pt")
            possible_names.add(f"{pdb}_{key}")
            possible_names.add(key + ".pt")
        # also store direct map by pdb+key tuple
        lig_map[(pdb, key)] = smi
        # map by names too
        for name in possible_names:
            lig_map[name] = smi

    # load activity CSV
    if not os.path.exists(activity_csv):
        print("Activity CSV not found:", activity_csv)
        print("No labels will be added. Provide --activity path to CSV with canonical_smiles and pchembl_value or standard_value+standard_units.")
        return

    act_df = pd.read_csv(activity_csv)
    act_map = build_activity_map(act_df)
    print("Activity map entries:", len(act_map))

    pair_files = sorted(glob.glob("data/pyg/dataset/pair_*.pt"))
    if not pair_files:
        print("No pair files found in data/pyg/dataset")
        return

    summary = []
    for pf in pair_files:
        d = torch.load(pf, weights_only=False)
        meta = d.get('meta', {}) or {}
        # try find ligand file name from meta
        candidate_smi = None
        # common meta styles: meta['ligand_file'] or meta['ligand'] or meta['pdb']
        lf = meta.get('ligand_file') or meta.get('ligand') or meta.get('ligand_pt') or meta.get('lig_file')
        # if lf is Data object, try to find its name in the ligdf by matching shapes (fallback)
        if isinstance(lf, str):
            # direct name lookup in lig_map
            candidate_smi = lig_map.get(lf)
            # also try basename
            if candidate_smi is None:
                candidate_smi = lig_map.get(os.path.basename(lf))
            # also try splitting underscore keys
            if candidate_smi is None and "_" in lf:
                tokens = lf.split("_")
                # try (pdb, key)
                if len(tokens) >= 2:
                    pdb = tokens[0]
                    key = "_".join(tokens[1:]).replace(".pt","")
                    candidate_smi = lig_map.get((pdb, key))
        # fallback: try to use pdb+any ligand_key in ligand_index
        if candidate_smi is None and 'pdb' in meta:
            pdb = meta['pdb']
            # search lig_map for entries with this pdb
            for (k1,k2), v in list(lig_map.items()):
                # some lig_map items are keyed by (pdb, key)
                if isinstance((k1,k2), tuple) and k1 == pdb and v:
                    candidate_smi = v
                    break
        label = None
        if candidate_smi:
            label = act_map.get(candidate_smi)
        # final fallback: check if meta already has y
        if label is None and 'y' in meta and not pd.isna(meta.get('y')):
            try:
                label = float(meta.get('y'))
            except:
                label = None

        if label is not None:
            if ('y' in meta) and (not force):
                note = "exists"
            else:
                meta['y'] = float(label)
                d['meta'] = meta
                torch.save(d, pf)
                note = "written"
        else:
            note = "missing"

        summary.append({'pair_file':pf, 'pdb':meta.get('pdb'), 'ligand_file': lf, 'smi': candidate_smi, 'label': label, 'note': note})

    outdf = pd.DataFrame(summary)
    outdf.to_csv("data/pyg/dataset_labeling_summary.csv", index=False)
    print("Wrote summary to data/pyg/dataset_labeling_summary.csv")
    print(outdf.note.value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity", default="data/activity/chembl_activity.csv", help="Activity CSV path")
    parser.add_argument("--ligand_index", default="data/ligands/ligand_index.csv", help="Ligand index CSV")
    parser.add_argument("--force", action="store_true", help="Overwrite existing meta['y']")
    args = parser.parse_args()
    main(args)

