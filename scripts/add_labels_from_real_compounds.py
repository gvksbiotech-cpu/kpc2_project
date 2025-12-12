#!/usr/bin/env python3
import os, glob, math, pandas as pd, torch
from pathlib import Path

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def compute_pic50_from_standard(val, units):
    if val is None or units is None:
        return None
    try:
        v = float(val)
    except:
        return None
    if v <= 0:
        return None
    u = str(units).strip().lower()
    if 'nm' in u:
        return 9.0 - math.log10(v)
    if 'um' in u or 'µm' in u:
        return 6.0 - math.log10(v)
    if 'mm' in u:
        return 3.0 - math.log10(v)
    if u in ('m','mol','molar'):
        return -math.log10(v)
    return None

# Load inputs
manual_map = pd.read_csv("data/manual_label_map.csv", dtype=str).fillna('')
activity = pd.read_csv("data/activity/true_compound_activity.csv", dtype=str).fillna('')

# Build activity lookup by query_name (the name used when fetching; also try molecule_chembl_id)
# We'll pick the best pchembl_value if present, else compute pIC50 from standard_value/units
act_map = {}
for _, r in activity.iterrows():
    q = str(r.get('query_name') or r.get('query') or '').strip()
    if not q:
        continue
    # prefer pchembl_value
    pch = safe_float(r.get('pchembl_value'))
    if pch is not None:
        val = pch
    else:
        val = compute_pic50_from_standard(r.get('standard_value'), r.get('standard_units'))
    if val is None:
        continue
    qkey = q.strip().lower()
    # choose highest activity if duplicates
    if qkey not in act_map or val > act_map[qkey]:
        act_map[qkey] = float(val)

print("Built activity map for", len(act_map), "compound names (keys sample):", list(act_map.keys())[:10])

# Now annotate pair files
pair_files = sorted(glob.glob("data/pyg/dataset/pair_*.pt"))
summary = []
for pf in pair_files:
    d = torch.load(pf, weights_only=False)
    meta = d.get('meta', {}) or {}
    pdb = meta.get('pdb') or ''
    # find mapping row for this pdb in manual_map
    mm = manual_map[manual_map['pdb'].astype(str).str.strip().str.upper() == str(pdb).strip().upper()]
    label = None
    note = 'no_map'
    if len(mm) > 0:
        real_name = str(mm.iloc[0]['real_compound_name']).strip()
        note = 'mapped'
        candidate = act_map.get(real_name.strip().lower())
        if candidate is None:
            # try fuzzy: try matching by substring keys
            for k in act_map.keys():
                if real_name.strip().lower() in k or k in real_name.strip().lower():
                    candidate = act_map[k]
                    note = 'mapped_by_substring'
                    break
        if candidate is not None:
            meta['y'] = float(candidate)
            d['meta'] = meta
            torch.save(d, pf)
            label = float(candidate)
            note = note + '_written'
        else:
            note = note + '_no_activity'
    else:
        note = 'no_manual_map'
    summary.append({'pair_file':pf, 'pdb':pdb, 'label':label, 'note':note})

# write summary
out = pd.DataFrame(summary)
os.makedirs("data/pyg", exist_ok=True)
out.to_csv("data/pyg/dataset_labeling_from_realcompounds.csv", index=False)
print("Wrote labeling summary to data/pyg/dataset_labeling_from_realcompounds.csv")
print(out['note'].value_counts())
