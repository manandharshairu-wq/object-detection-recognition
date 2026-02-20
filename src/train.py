from typing import Dict, List, Tuple
import torch

from src.utils import AverageMeter, Timer, save_checkpoint

def train_one_epoch(model, loader, optimizer, device, scaler=None) -> float:
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), n=len(images))

    return float(losses.avg)

@torch.inference_mode()
def val_loss(model, loader, device) -> float:
    # torchvision detection returns losses only in train mode
    model.train()
    losses = AverageMeter()

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        losses.update(loss.item(), n=len(images))

    return float(losses.avg)

def fit_frcnn(
    model,
    train_loader,
    val_loader,
    device: str,
    epochs: int,
    patience: int,
    ckpt_path,
    lr: float = 1e-4,
    mixed_precision: bool = True,
):
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.cuda.amp.GradScaler() if (mixed_precision and device.type == "cuda") else None

    best = float("inf")
    bad = 0
    history: List[Dict[str, float]] = []

    with Timer() as t_train:
        for epoch in range(1, epochs + 1):
            tr = train_one_epoch(model, train_loader, optimizer, device, scaler)
            va = val_loss(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})

            # checkpoint best
            if va < best - 1e-4:
                best = va
                bad = 0
                save_checkpoint(ckpt_path, epoch, model, optimizer, extra={"best_val_loss": best})
            else:
                bad += 1
                if bad >= patience:
                    break

    return model, history, t_train.seconds
