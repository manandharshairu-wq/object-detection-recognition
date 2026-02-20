from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Project paths (assignment2/)
    root: Path = Path(".")
    data_dir: Path = root / "data"
    outputs_dir: Path = root / "outputs"
    ckpt_dir: Path = root / "checkpoints"
    splits_dir: Path = root / "splits"

    # Assignment requirements
    img_size: int = 512
    split_train: float = 0.70
    split_val: float = 0.15
    split_test: float = 0.15

    # Batch sizes (as recommended)
    frcnn_batch_size: int = 2     # 2–4
    yolo_batch_size: int = 8      # 8–16
    num_workers: int = 0

    # Epochs (within required ranges)
    penn_epochs: int = 12         # 10–15
    pets_epochs: int = 18         # 15–20

    # Early stopping
    patience: int = 3

    # Seed
    seed: int = 42
