import torch
from tqdm import tqdm
from datasets import training_mask, inference_mask, Metrics
from models import MaskVatAdaLN

import numpy as np

class AverageMeter:
    """
    Tracks and computes the running average of a scalar metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    """
    Computes top-k accuracy for the given logits and targets.

    Args:
        logits (Tensor): Model outputs of shape (B, C)
        targets (Tensor): Ground-truth labels of shape (B,)
        topk (tuple): Values of k for top-k accuracy

    Returns:
        List[Tensor]: Accuracy values in percentage for each k
    """
    maxk = min(max(topk), logits.size(1))
    batch_size = targets.size(0)

    _, preds = logits.topk(maxk, dim=1, largest=True, sorted=True)
    preds = preds.t()
    correct = preds.eq(targets.view(1, -1))

    accuracies = []
    for k in topk:
        k = min(k, maxk)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        accuracies.append(correct_k * 100.0 / batch_size)

    return accuracies


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    scaler,
    device,
    criterion,
    codebooks_size=1024,
):
    """
    Runs one training epoch.

    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training dataloader
        optimizer (Optimizer): Optimizer
        scheduler (LRScheduler or None): Learning rate scheduler
        scaler (GradScaler): AMP gradient scaler
        device (torch.device): Training device
        criterion (callable): Loss function

    Returns:
        tuple: (avg_accuracy, avg_loss)
    """
    model.train()

    loss_meter = AverageMeter()
    # acc_meter = AverageMeter()

    progress = tqdm(
        dataloader,
        desc="Train",
        dynamic_ncols=True,
        leave=False,
    )

    for encodings in progress:
        optimizer.zero_grad(set_to_none=True)

        # Move data to device
        for i in range(len(encodings)):
            encodings[i] = encodings[i].to(device, non_blocking=True)

        dac_encoding, clip_encoding, s3d_encoding = encodings
        masked_encodings, targets = training_mask(dac_encoding, codebooks_size)

        # Forward pass (mixed precision)
        with torch.amp.autocast(device_type="cuda"):
            outputs = model(masked_encodings, clip_encoding, s3d_encoding)
            logits = torch.permute(outputs, (0, 3, 1, 2))
            loss = criterion(logits, targets)

        batch_loss = loss.item()
        if np.isnan(batch_loss).any() or np.isinf(batch_loss).any():
            continue

        # Backward + optimizer step (AMP-safe)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        # Metrics

        loss_meter.update(batch_loss)

        # TODO: maybe use topk cosine similarity as extra metric
        # with torch.no_grad():
        #     batch_acc = topk_accuracy(logits, dac_encoding, topk=(1,))[0].item()
        #     acc_meter.update(batch_acc)

        #track memory usage
        current_mem = torch.cuda.memory_allocated() / (1024 ** 3) # Convert to GB
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        # Progress bar update
        progress.set_postfix(
            loss=f"{batch_loss:.4f} ({loss_meter.avg:.4f})",
            # acc=f"{batch_acc:.2f}% ({acc_meter.avg:.2f}%)",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            mem=f"{current_mem:.1f}GB (Peak: {peak_mem:.1f}GB)"
        )

    # Step scheduler once per epoch (for epoch-based schedulers)
    if scheduler is not None:
        scheduler.step()

    # return acc_meter.avg, loss_meter.avg
    return loss_meter.avg


@torch.no_grad()
def valid_epoch(
    model: MaskVatAdaLN,
    dataloader,
    device,
    metrics: Metrics,
    steps=20,
    codebook_size=1024,
):
    # TODO: test with metrics
    model.eval()

    distance_meter = AverageMeter()
    # waveclip_meter = AverageMeter()

    progress = tqdm(
        dataloader,
        desc="Val (Cls)",
        dynamic_ncols=True,
        leave=False,
    )

    for encodings in progress:
        # Move data to device
        for i in range(len(encodings)):
            encodings[i] = encodings[i].to(device, non_blocking=True)

        dac_encoding, clip_encoding, s3d_encoding = encodings
        masked_encodings = torch.full(dac_encoding.shape, codebook_size, device=device)

        for step in range(steps):
            # Forward pass (inference-only)
            outputs = model(masked_encodings, clip_encoding, s3d_encoding)
            masked_encodings = inference_mask(outputs, step, steps)

        # Metrics
        predictions = model.decode(masked_encodings)
        targets = model.decode(dac_encoding)
        results = metrics.get_metrics(
            predictions, targets, clip_encoding, FDM=True
        )
        batch_fdm = results["FDM"]
        # batch_cos = results["cos"]
        distance_meter.update(batch_fdm)
        # waveclip_meter.update(batch_cos)

        # Progress bar update
        progress.set_postfix(
            distance=f"{batch_fdm:.4f} ({distance_meter.avg:.4f})",
            # waveclip=f"{batch_cos:.2f}% ({waveclip_meter.avg:.2f}%)",
        )

    return distance_meter.avg # , waveclip_meter.avg
