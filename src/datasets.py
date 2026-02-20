from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import tarfile
import urllib.request
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from lxml import etree
import torchvision.transforms.functional as F


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ---------------------------
# Penn-Fudan (manual download)
# ---------------------------
PENNFUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

def prepare_pennfudan(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "PennFudanPed.zip"
    data_dir = root / "PennFudanPed"
    if not data_dir.exists():
        if not zip_path.exists():
            urllib.request.urlretrieve(PENNFUDAN_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)
    return data_dir

def masks_to_boxes(mask: torch.Tensor) -> torch.Tensor:
    # mask: [H,W] int, 0 background, 1..N instance ids
    obj_ids = torch.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]

    boxes = []
    for oid in obj_ids:
        ys, xs = torch.where(mask == oid)
        if ys.numel() == 0:
            continue
        x1 = xs.min().item()
        y1 = ys.min().item()
        x2 = xs.max().item()
        y2 = ys.max().item()
        boxes.append([x1, y1, x2, y2])

    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)

class PennFudanDetection(Dataset):
    def __init__(self, root: str, img_size: int = 512):
        self.root = Path(root)
        data_dir = prepare_pennfudan(self.root)

        self.img_dir = data_dir / "PNGImages"
        self.mask_dir = data_dir / "PedMasks"

        self.images = sorted(self.img_dir.glob("*.png"))
        self.masks = sorted(self.mask_dir.glob("*.png"))
        assert len(self.images) == len(self.masks)

        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img_pil = Image.open(img_path).convert("RGB")
        mask_pil = Image.open(mask_path)

        w, h = img_pil.size

        # image -> tensor
        img = F.pil_to_tensor(img_pil).float() / 255.0  # [3,H,W]

        # mask -> numpy -> tensor (NO None anywhere)
        mask_np = np.array(mask_pil, dtype=np.int64)     # [H,W]
        mask = torch.from_numpy(mask_np)                 # int64

        boxes = masks_to_boxes(mask)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # class 1 = person

        # resize image
        img = F.resize(img, [self.img_size, self.img_size])

        # scale boxes
        sx = self.img_size / w
        sy = self.img_size / h
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        target = {"boxes": boxes, "labels": labels}
        return img, target


# ---------------------------
# Oxford-IIIT Pets subset
# ---------------------------
PETS_IMG_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
PETS_ANN_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

def _download_and_extract(url: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / url.split("/")[-1]
    if not fname.exists():
        urllib.request.urlretrieve(url, fname)
    with tarfile.open(fname, "r:gz") as tar:
        tar.extractall(path=out_dir)

def prepare_pets(root: Path) -> Tuple[Path, Path]:
    _download_and_extract(PETS_IMG_URL, root)
    _download_and_extract(PETS_ANN_URL, root)
    return root / "images", root / "annotations" / "xmls"

def parse_bbox(xml_path: Path) -> Tuple[int, int, int, int]:
    tree = etree.parse(str(xml_path))
    bbox = tree.find(".//bndbox")
    xmin = int(bbox.findtext("xmin"))
    ymin = int(bbox.findtext("ymin"))
    xmax = int(bbox.findtext("xmax"))
    ymax = int(bbox.findtext("ymax"))
    return xmin, ymin, xmax, ymax

class PetsSubsetDetection(Dataset):
    def __init__(self, root: str, breeds: List[str], img_size: int = 512):
        self.root = Path(root)
        images_dir, xmls_dir = prepare_pets(self.root)

        self.items = []
        for jpg in images_dir.glob("*.jpg"):
            stem = jpg.stem
            breed_prefix = "_".join(stem.split("_")[:-1])
            if breed_prefix in breeds:
                xml = xmls_dir / f"{stem}.xml"
                if xml.exists():
                    self.items.append((jpg, xml, breed_prefix))

        self.breeds = breeds
        self.label_map = {b: i + 1 for i, b in enumerate(breeds)}
        self.img_size = img_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        jpg_path, xml_path, breed = self.items[idx]
        img_pil = Image.open(jpg_path).convert("RGB")
        w, h = img_pil.size

        img = F.pil_to_tensor(img_pil).float() / 255.0
        img = F.resize(img, [self.img_size, self.img_size])

        xmin, ymin, xmax, ymax = parse_bbox(xml_path)
        sx = self.img_size / w
        sy = self.img_size / h

        box = torch.tensor([[xmin*sx, ymin*sy, xmax*sx, ymax*sy]], dtype=torch.float32)
        label = torch.tensor([self.label_map[breed]], dtype=torch.int64)

        target = {"boxes": box, "labels": label}
        return img, target
