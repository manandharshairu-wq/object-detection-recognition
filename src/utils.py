import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# ---- Reproducibility ----
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---- AverageMeter (like sgrvinod's training loop style) ----
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

# ---- Timing context manager ----
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.t1 = time.perf_counter()
        self.seconds = self.t1 - self.t0

# ---- 70/15/15 split ----
def split_indices(n: int, train: float, val: float, test: float, seed: int) -> Dict[str, List[int]]:
    assert abs((train + val + test) - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train)
    n_val = int(n * val)

    return {
        "train": idx[:n_train].tolist(),
        "val": idx[n_train:n_train+n_val].tolist(),
        "test": idx[n_train+n_val:].tolist(),
    }

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def load_json(path: Path) -> Any:
    return json.loads(path.read_text())

# ---- Checkpointing (sgrvinod-style) ----
def save_checkpoint(path: Path, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer, extra: Dict[str, Any] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
