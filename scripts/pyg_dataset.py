
# Begin: allowlist torch-geometric Data classes for torch.load unpickling (safe for local files)
try:
    import torch_geometric
    from torch_geometric.data import Data as _TorchGeoData
    # DataEdgeAttr may be required depending on PyG version — try to import it
    try:
        from torch_geometric.data import DataEdgeAttr as _TorchGeoDataEdgeAttr
        torch.serialization.add_safe_globals([_TorchGeoData, _TorchGeoDataEdgeAttr])
    except Exception:
        # fallback: only add Data
        torch.serialization.add_safe_globals([_TorchGeoData])
except Exception:
    pass
# End allowlist


# scripts/pyg_dataset.py
import os, glob, torch
from torch_geometric.data import Dataset
import torch.nn.functional as F

DATA_DIR = "data/pyg/dataset"

class PocketLigandPairsDataset(Dataset):
    def __init__(self, root=DATA_DIR, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.files = sorted(glob.glob(os.path.join(root, "pair_*.pt")))

    def len(self):
        return len(self.files)

    def get(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        # structure: {'pocket':Data, 'ligand':Data, 'meta':{...}}
        pocket = d['pocket']
        ligand = d['ligand']
        meta = d.get('meta', {})
        # ensure tensors are present
        return {"pocket": pocket, "ligand": ligand, "meta": meta}
