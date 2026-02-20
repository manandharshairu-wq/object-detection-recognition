from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import torch
from PIL import Image

def xyxy_to_yolo(box_xyxy: torch.Tensor, w: int, h: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box_xyxy.tolist()
    xc = ((x1 + x2) / 2.0) / w
    yc = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh

def export_to_yolo(dataset, split: Dict[str, List[int]], out_root: Path, class_names: List[str]) -> Path:
    """
    dataset[idx] -> (img_tensor [3,512,512] float 0..1, target dict boxes xyxy, labels 1..K)
    YOLO labels must be 0..K-1
    """
    out_root.mkdir(parents=True, exist_ok=True)

    for s in ["train", "val", "test"]:
        (out_root / "images" / s).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / s).mkdir(parents=True, exist_ok=True)

        for new_i, idx in enumerate(split[s]):
            img, target = dataset[idx]
            H, W = img.shape[-2], img.shape[-1]

            img_pil = Image.fromarray((img.permute(1,2,0).numpy() * 255).astype("uint8"))
            img_path = out_root / "images" / s / f"{new_i:06d}.jpg"
            img_pil.save(img_path, quality=95)

            label_path = out_root / "labels" / s / f"{new_i:06d}.txt"
            lines = []
            for box, lab in zip(target["boxes"], target["labels"]):
                cls = int(lab.item()) - 1
                xc, yc, bw, bh = xyxy_to_yolo(box, W, H)
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            label_path.write_text("\n".join(lines))

    data_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return yaml_path
