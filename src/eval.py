from typing import Dict, List
import torch

from src.metrics import map50, precision_recall_iou50
from src.utils import Timer

@torch.inference_mode()
def eval_frcnn(model, test_loader, device: str) -> Dict[str, float]:
    device = torch.device(device)
    model.to(device)
    model.eval()

    preds: List[Dict[str, torch.Tensor]] = []
    targets: List[Dict[str, torch.Tensor]] = []

    for images, targs in test_loader:
        images = [img.to(device) for img in images]
        out = model(images)[0]
        preds.append({k: v.detach().cpu() for k, v in out.items()})
        targets.append({k: v.detach().cpu() for k, v in targs[0].items()})

    m = map50(preds, targets)
    p, r = precision_recall_iou50(preds, targets, score_thresh=0.5)

    # Inference speed (images/sec), measure on up to 50 samples
    n_measure = min(50, len(test_loader.dataset))
    # warmup
    dummy = torch.zeros((3, 512, 512), device=device)
    _ = model([dummy])
    _ = model([dummy])

    with Timer() as t:
        for i in range(n_measure):
            img, _ = test_loader.dataset[i]
            _ = model([img.to(device)])
    ips = n_measure / max(t.seconds, 1e-9)

    return {
        "mAP@0.5": float(m),
        "precision": float(p),
        "recall": float(r),
        "inference_img_per_sec": float(ips),
    }
