from typing import Dict, List, Tuple
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

def _iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [N,4], b: [M,4], xyxy
    N, M = a.size(0), b.size(0)
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=a.device)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ix1 = torch.maximum(ax1, bx1)
    iy1 = torch.maximum(ay1, by1)
    ix2 = torch.minimum(ax2, bx2)
    iy2 = torch.minimum(ay2, by2)

    iw = torch.clamp(ix2 - ix1, min=0)
    ih = torch.clamp(iy2 - iy1, min=0)
    inter = iw * ih

    area_a = torch.clamp(ax2 - ax1, min=0) * torch.clamp(ay2 - ay1, min=0)
    area_b = torch.clamp(bx2 - bx1, min=0) * torch.clamp(by2 - by1, min=0)
    union = area_a + area_b - inter
    return inter / torch.clamp(union, min=1e-6)

@torch.inference_mode()
def map50(preds: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> float:
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    metric.update(preds, targets)
    out = metric.compute()
    return float(out["map"].item())  # with iou_thresholds=[0.5], this is mAP@0.5

def precision_recall_iou50(preds, targets, score_thresh: float = 0.5) -> Tuple[float, float]:
    tp = fp = fn = 0
    for p, t in zip(preds, targets):
        p_boxes = p["boxes"]
        p_scores = p["scores"]
        p_labels = p["labels"]

        keep = p_scores >= score_thresh
        p_boxes = p_boxes[keep]
        p_labels = p_labels[keep]
        p_scores = p_scores[keep]

        if p_scores.numel() > 0:
            order = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[order]
            p_labels = p_labels[order]

        t_boxes = t["boxes"]
        t_labels = t["labels"]
        matched = set()

        for i in range(p_boxes.size(0)):
            same = (t_labels == p_labels[i]).nonzero(as_tuple=False).view(-1)
            same = [j.item() for j in same if j.item() not in matched]
            if not same:
                fp += 1
                continue
            ious = _iou_xyxy(p_boxes[i].unsqueeze(0), t_boxes[same]).squeeze(0)
            best = torch.argmax(ious).item()
            if ious[best].item() >= 0.5:
                tp += 1
                matched.add(same[best])
            else:
                fp += 1

        fn += (t_boxes.size(0) - len(matched))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return float(precision), float(recall)
