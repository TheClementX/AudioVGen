import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
import gc

import os

from models import MaskVatAdaLN
from datasets import get_datasets
from training import train_epoch, valid_epoch
from datasets import Metrics

from torchinfo import summary

import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", DEVICE)

config = {
    "seq_len": 862,  # seq_len of DAC encoding
    "embed_dim": 512,  # embed_dim for the model
    "n_heads": 16,  # number of heads for multiheaded attention
    "c_dim": 512,  # dimensions of clip encoding
    "s_dim": 1024,  # dimensions of s3d encoding
    "M": 12,  # number of AdaLN Blocks in the model
    "K": 9,  # dimensions of DAC encoding
    "codebook_size": 1024,  # size of codebook
    "weight_decay": 0.00001,  # from paper
    "lr": 0.0001,
    "epochs": 50,
    "data_root": "./VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video",  # root of audio-video data
    "batch_size": 16,  # batch size for training
    "checkpoint_dir": "./checkpoints",
}

# Datasets
train_dataset, valid_dataset = get_datasets(config["data_root"])

# Dataloaders
train_loader = DataLoader(
    dataset=train_dataset,
    num_workers=5,
    batch_size=config["batch_size"],
    pin_memory=True,
    shuffle=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    num_workers=5,
    batch_size=config["batch_size"],
    pin_memory=True,
)

# [dac, clip, s3d]
# print dataset metainfo
retlist = train_dataset.__getitem__(0)
print(f"dac dimensions {retlist[0].shape}")
print(f"clip dimensions {retlist[1].shape}")
print(f"s3d dimensions {retlist[2].shape}")

model = MaskVatAdaLN(
    config["seq_len"],
    config["embed_dim"],
    config["n_heads"],
    config["c_dim"],
    config["s_dim"],
    config["M"],
    config["K"],
    config["codebook_size"],
).to(DEVICE)

summary(model)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.AdamW(
    model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
)

# Scheduler
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=config['lr'], 
    epochs=config["epochs"],
    steps_per_epoch=len(train_loader)
)

# Mixed-Precision Training
scaler = torch.amp.GradScaler(device="cuda")

wandb.login(key="")

run_name = "inference_mask_correct"

run = wandb.init(
    name = run_name,
    entity = "AudioVGen",
    project = "AudioVGen_plots",
    config = config,
)

# Ensure checkpoint directory exists
os.makedirs(config["checkpoint_dir"], exist_ok=True)

def save_model(model, optimizer, scheduler, metrics, epoch, path):
    """
    Saves model, optimizer, scheduler, and training metrics to a checkpoint.

    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer
        scheduler (LRScheduler): Learning rate scheduler
        metrics (dict): Dictionary of tracked metrics
        epoch (int): Current epoch
        path (str): Path to save checkpoint
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "epoch": epoch,
        },
        path,
    )
    print(f"Checkpoint saved at {path}")


def load_model(
    model, optimizer=None, scheduler=None, path="./checkpoint.pth", device=None
):
    """
    Loads model, optimizer, scheduler, and metrics from a checkpoint.

    Args:
        model (nn.Module): Model to load weights into
        optimizer (Optimizer, optional): Optimizer to load state
        scheduler (LRScheduler, optional): Scheduler to load state
        path (str): Path to checkpoint file
        device (torch.device, optional): Device mapping for checkpoint

    Returns:
        tuple: model, optimizer, scheduler, epoch, metrics
    """
    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return model, optimizer, scheduler, epoch, metrics


start_epoch = 0
best_distance = np.inf
# best_cos_sim = 0.0
eval_cls = True

inference_metrics = Metrics()

gc.collect()
torch.cuda.empty_cache()

# torch.autograd.set_detect_anomaly(True)
for epoch in range(start_epoch, config["epochs"]):
    print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

    # -----------------------------
    # Train
    # -----------------------------
    train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, scaler, DEVICE, criterion
    )
    curr_lr = optimizer.param_groups[0]["lr"]
    print(f"Train | Loss: {train_loss:.4f} | LR: {curr_lr:.6f}")

    metrics = {
        "train_loss": train_loss,
        "lr": curr_lr,
    }

    # -----------------------------
    # Validation and compute frechet distance
    # -----------------------------
    if eval_cls:
        FDM, FDD, FAD = valid_epoch(
            model,
            valid_loader,
            DEVICE,
            inference_metrics,
        )
        print(f"Val (Cls) | Distance: {FDM:.4f}")
        metrics.update(
            {
                "FDM": FDM,
            }
        )

    # -----------------------------
    # Save checkpoints
    # -----------------------------
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"last_{run_name}.pth")
    save_model(model, optimizer, scheduler, metrics, epoch, checkpoint_path)
    print(f"Saved last epoch model: {checkpoint_path}")

    # Save model with best validation FDM distance
    if eval_cls and best_distance >= FDM:
        best_distance = FDM
        best_distance_path = os.path.join(config["checkpoint_dir"], "best_distance.pth")
        save_model(model, optimizer, scheduler, metrics, epoch, best_distance_path)
        # if "wandb" in globals() and run is not None:
        #     wandb.save(best_distance_path)
        print(f"Saved best distance validation model: {best_distance_path}")

    # Save model with best validation loss
    # if eval_cls and best_cos_sim <= cos_sim:
    #     best_cos_sim = cos_sim
    #     best_cos_sim_path = os.path.join(config["checkpoint_dir"], "best_cos_sim.pth")
    #     save_model(model, optimizer, scheduler, metrics, epoch, best_cos_sim_path)
    #     if "wandb" in globals() and run is not None:
    #         wandb.save(best_cos_sim_path)
    #     print(f"Saved best waveclip validation model: {best_cos_sim_path}")

    # -----------------------------
    # Log metrics
    # -----------------------------
    if "run" in globals() and run is not None:
        run.log(metrics)
