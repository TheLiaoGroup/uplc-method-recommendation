import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.utils import from_smiles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from transformers import AutoTokenizer


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def _split_indices(
    n_samples: int,
    train_ratio: float,
    shuffle: bool,
    seed: int
):
    """
    返回:
        train_idx, test_idx
    其中某一个可能为 None，表示不使用该集合
    """

    indices = np.arange(n_samples)

    # ===== 情况 1：全训练 =====
    if train_ratio == 1:
        return indices.tolist(), None

    # ===== 情况 2：全测试 =====
    if train_ratio == 0:
        return None, indices.tolist()

    # ===== 情况 3：正常 split =====
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in [0, 1], got {train_ratio}")

    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_ratio,
        shuffle=shuffle,
        random_state=seed
    )

    return train_idx.tolist(), test_idx.tolist()
    
def build_quantile_bins_and_weights(y_train, n_bins=10, sqrt_inv=True, max_weight=4.0):
    import numpy as np
    import torch

    if isinstance(y_train, torch.Tensor):
        y_np = y_train.detach().cpu().numpy().reshape(-1)
    else:
        y_np = np.asarray(y_train).reshape(-1)

    # 分位数边界（避免某些 bin 为空）
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_np, qs)

    # 极端情况下会有重复边界（比如大量相同值），做去重
    edges = np.unique(edges)
    if len(edges) < 3:
        # 至少需要：min < ... < max
        raise ValueError("Not enough unique y values to build bins.")

    # 给每个样本分 bin：0...(num_bins-1)
    bin_ids = np.digitize(y_np, edges[1:-1], right=False)
    freq = np.bincount(bin_ids, minlength=len(edges) - 1)

    inv = 1.0 / (freq + 1e-6)
    if sqrt_inv:
        inv = np.sqrt(inv)

    inv = inv / (inv.mean() + 1e-12)     # 平均权重归一到 1
    inv = np.clip(inv, 0.0, max_weight)  # 截断防爆

    bin_edges = torch.tensor(edges, dtype=torch.float32)
    bin_weights = torch.tensor(inv, dtype=torch.float32)
    return bin_edges, bin_weights

class GNNDataset(Dataset):
    def __init__(self, smiles, y, phys=None):
        self.smiles = smiles
        self.y_scaler = StandardScaler()
        self.y = torch.tensor(
            self.y_scaler.fit_transform(np.array(y).reshape(-1, 1)).squeeze(),
            dtype=torch.float32)

        self.phys = None
        if phys is not None:
            self.phys_scaler = StandardScaler()
            self.phys = self.phys_scaler.fit_transform(phys)

        self.graphs = self._build_graphs()

        self.bin_edges, self.bin_weights = build_quantile_bins_and_weights(
            self.y, n_bins=10, sqrt_inv=True, max_weight=2.0)

    def _build_graphs(self):
        graphs = []
        for i, smi in enumerate(self.smiles):
            try:
                g = from_smiles(smi)
                if g is None or g.x is None:
                    continue
                g.x = g.x.float()
                if self.phys is not None:
                    extra = torch.tensor(self.phys[i], dtype=torch.float32)
                    g.x = torch.cat(
                        [g.x, extra.expand(g.num_nodes, -1)], dim=1
                    )
                g.y = self.y[i].unsqueeze(0)
                g.smiles = smi
                graphs.append(g)
            except Exception:
                continue
        return graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @property
    def rt_scaler(self):
        return self.y_scaler

    def get_feature_dim(self):
        if len(self.graphs) == 0:
            return 0
        return self.graphs[0].x.shape[1]

class BERTDataset(Dataset):
    def __init__(
        self, smiles, y, tokenizer,
        max_length=128, phys=None, augment=False):

        self.smiles = smiles
        self.augment = augment
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._phys_dim = 0

        self.y_scaler = StandardScaler()
        self.y = torch.tensor(
            self.y_scaler.fit_transform(np.array(y).reshape(-1, 1)).squeeze(),
            dtype=torch.float32)

        self.phys = None
        if phys is not None:
            self.phys_scaler = StandardScaler()
            self.phys = self.phys_scaler.fit_transform(phys)
            self._phys_dim = self.phys.shape[1]

        self.bin_edges, self.bin_weights = build_quantile_bins_and_weights(
            self.y, n_bins=10, sqrt_inv=True, max_weight=4.0)

    def _rand_smiles(self, s):
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, doRandom=True) if mol else s

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.augment:
            smi = self._rand_smiles(smi)

        enc = self.tokenizer(
            smi,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self.y[idx].unsqueeze(0)

        if self.phys is not None:
            item["physicochemical"] = torch.tensor(
                self.phys[idx], dtype=torch.float32
            )

        return item

    @property
    def phys_dim(self):
        return self._phys_dim

    @property
    def rt_scaler(self):
        return self.y_scaler



def create_data_loaders(
    smiles, y, phys=None,
    model_type="gnn", train_ratio=0.8,
    shuffle=False, args=None):

    gen = torch.Generator()
    gen.manual_seed(args.seed)


    train_idx, test_idx = _split_indices(
        len(smiles), train_ratio, shuffle, args.seed)

    def _subset(arr, idx):
        return [arr[i] for i in idx] if idx is not None else arr

    if model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)

        train_ds = BERTDataset(
            _subset(smiles, train_idx),
            _subset(y, train_idx),
            tokenizer,
            max_length=args.max_seq_length,
            phys=_subset(phys, train_idx) if phys is not None else None,
            augment=args.augment)

        test_ds = BERTDataset(
            _subset(smiles, test_idx),
            _subset(y, test_idx),
            tokenizer,
            max_length=args.max_seq_length,
            phys=_subset(phys, test_idx) if phys is not None else None,
            augment=False)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            generator=gen)
        
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            generator=gen)

    elif model_type == "gnn":
        train_ds = GNNDataset(
            _subset(smiles, train_idx),
            _subset(y, train_idx),
            phys=_subset(phys, train_idx) if phys is not None else None)
        
        test_ds = GNNDataset(
            _subset(smiles, test_idx),
            _subset(y, test_idx),
            phys=_subset(phys, test_idx) if phys is not None else None)

        train_loader = GeoDataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            worker_init_fn=_worker_init_fn,
            generator=gen)
        
        test_loader = GeoDataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            generator=gen)

    return train_loader, test_loader, train_ds, test_ds
