from pathlib import Path
from typing import Dict
import json

from ultralytics import YOLO

def run_yolov8n(data_yaml: Path, out_dir: Path, exp_name: str, epochs: int, imgsz: int, batch: int) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")  # pretrained nano model (transfer learning)

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(out_dir),
        name=exp_name,
        verbose=False,
    )

    val = model.val(data=str(data_yaml), imgsz=imgsz, batch=1, verbose=False)

    results = {
        "mAP@0.5": float(val.box.map50),
        "precision": float(val.box.mp),
        "recall": float(val.box.mr),
        "speed_ms_per_img": dict(val.speed),  # preprocess/inference/postprocess
    }
    return results
