#!/usr/bin/env python3
"""
Robust trainer that loads prepared pair_*.pt files and runs a tiny GNN smoke-test.
This script avoids unpickling issues by using torch.load(..., weights_only=False)
and resolves any filename strings inside the pair meta into real Data objects.
"""
import os, glob, torch, random
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Utilities
# -------------------------
def tload(path):
    """Safe load of local .pt files containing Data objects"""
    return torch.load(path, weights_only=False)

def resolve_data(obj):
    """If obj is a torch_geometric Data object return it.
       If obj is a string filename, try to load referenced .pt file.
       Otherwise return obj unchanged.
    """
    try:
        from torch_geometric.data import Data as TGData
    except Exception:
        TGData = None
    if TGData is not None and isinstance(obj, TGData):
        return obj
    if isinstance(obj, str):
        # try as-is
        if os.path.exists(obj):
            return tload(obj)
        # try in pockets/ligands dirs
        base = os.path.basename(obj)
        for d in ("data/pyg/pockets_atom", "data/pyg/ligands"):
            cand = os.path.join(d, base)
            if os.path.exists(cand):
                return tload(cand)
        stem = os.path.splitext(base)[0]
        for d in ("data/pyg/pockets_atom", "data/pyg/ligands"):
            for f in glob.glob(os.path.join(d, "*"+stem+"*.pt")):
                try:
                    return tload(f)
                except Exception:
                    pass
    return obj

# -------------------------
# Load pair files into memory (small dataset)
# -------------------------
pair_files = sorted(glob.glob("data/pyg/dataset/pair_*.pt"))
if len(pair_files) == 0:
    print("No pair files found in data/pyg/dataset. Run the pairing script first.")
    raise SystemExit(1)

pairs = []
for pf in pair_files:
    d = tload(pf)
    pocket = resolve_data(d.get("pocket"))
    ligand = resolve_data(d.get("ligand"))
    meta = d.get("meta", {}) or {}
    pairs.append((pocket, ligand, meta))

print("Loaded", len(pairs), "pairs. Example meta:", pairs[0][2])

# -------------------------
# Minimal PyG model (small)
# -------------------------
class SimpleGNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        if batch is None:
            # global pool by averaging nodes
            return x.mean(dim=0, keepdim=True)
        else:
            return global_mean_pool(x, batch)

class PocketLigandModel(nn.Module):
    def __init__(self, p_in, l_in, hidden=64):
        super().__init__()
        self.penc = SimpleGNNEncoder(p_in, hidden, hidden)
        self.lenc = SimpleGNNEncoder(l_in, hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, pocket, ligand):
        # pocket, ligand are Data objects
        px = pocket.x.to(device)
        pedge = pocket.edge_index.to(device)
        # create fake batch (all nodes belong to batch 0)
        pbatch = torch.zeros(px.size(0), dtype=torch.long, device=device)
        lx = ligand.x.to(device)
        ledge = ligand.edge_index.to(device)
        lbatch = torch.zeros(lx.size(0), dtype=torch.long, device=device)
        p_emb = self.penc(px, pedge, pbatch)
        l_emb = self.lenc(lx, ledge, lbatch)
        cat = torch.cat([p_emb, l_emb], dim=1)
        out = self.head(cat)
        return out.squeeze(1)

# -------------------------
# Prepare dims (infer from first pair)
# -------------------------
# find first valid pair with numeric x dims
p_dim = None; l_dim = None
for pocket, ligand, meta in pairs:
    try:
        p_dim = pocket.x.shape[1]
        l_dim = ligand.x.shape[1]
        break
    except Exception:
        continue

if p_dim is None or l_dim is None:
    print("Could not infer node feature dims from pairs. Inspect pocket/ligand Data objects.")
    raise SystemExit(1)

# -------------------------
# Training loop (tiny)
# -------------------------
model = PocketLigandModel(p_dim, l_dim).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def meta_to_label(meta):
    """Return standardized label (z = (y-mean)/std). If no stats file, use raw y or 0.0."""
    import os, pandas as _pd
    # load mean/std if available
    mean, std = 0.0, 1.0
    stats_path = 'data/pyg/label_stats.csv'
    if os.path.exists(stats_path):
        try:
            st = _pd.read_csv(stats_path).iloc[0].to_dict()
            mean = float(st.get('mean', 0.0)) if st.get('mean', None) is not None else 0.0
            std = float(st.get('std', 1.0)) if st.get('std', None) is not None and float(st.get('std',1.0))>0 else 1.0
        except Exception:
            mean, std = 0.0, 1.0

    # extract numeric y
    if meta is None:
        return 0.0
    if isinstance(meta, dict):
        v = meta.get('y', None)
        if v is None:
            return 0.0
        try:
            y = float(v)
        except:
            return 0.0
    else:
        try:
            y = float(meta)
        except:
            return 0.0

    # standardize and return
    return (y - mean) / std


# --- clean train/validation loop inserted by assistant ---
random.shuffle(pairs)
n = len(pairs); split = int(0.8 * n)
train_pairs = pairs[:split]
val_pairs = pairs[split:]

epochs = 5
for ep in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for pocket, ligand, meta in train_pairs:
        # ensure Data objects
        pocket = resolve_data(pocket)
        ligand = resolve_data(ligand)
        from torch_geometric.data import Data as TGData
        if not isinstance(pocket, TGData) or not isinstance(ligand, TGData):
            continue
        opt.zero_grad()
        y = torch.tensor([meta_to_label(meta)], device=device, dtype=torch.float32)
        out = model(pocket, ligand)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss / max(1, len(train_pairs))
    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for pocket, ligand, meta in val_pairs:
            pocket = resolve_data(pocket)
            ligand = resolve_data(ligand)
            from torch_geometric.data import Data as TGData
            if not isinstance(pocket, TGData) or not isinstance(ligand, TGData):
                continue
            y = torch.tensor([meta_to_label(meta)], device=device, dtype=torch.float32)
            out = model(pocket, ligand)
            val_losses.append(((out - y)**2).item())
    val_rmse = (sum(val_losses)/max(1,len(val_losses)))**0.5
    print(f"Epoch {ep} train_loss: {avg_loss:.4f} val_RMSE: {val_rmse:.4f}")

print("Training finished; saving model to models/simple_pl_model.pt")
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/simple_pl_model.pt")
# --- end inserted block ---

