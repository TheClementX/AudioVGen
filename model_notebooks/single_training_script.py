import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
import gc

import os
from configparser import ConfigParser

from mamba import AudioVGen
from datasets import get_datasets
from training import train_epoch, valid_epoch
from datasets import Metrics

from torchinfo import summary
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import numpy as np
import dac

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", DEVICE)

config = {
    "seq_len": 862,  # seq_len of DAC encoding
    "embed_dim": 512,  # embed_dim for the model
    "n_heads": 16,  # number of heads for multiheaded attention
    "d_state": 128, 
    "d_conv": 4, 
    "expand": 2, 
    "c_dim": 512,  # dimensions of clip encoding
    "s_dim": 1024,  # dimensions of s3d encoding
    "M": 12,  # number of AdaLN Blocks in the model
    "K": 9,  # dimensions of DAC encoding
    "codebook_size": 1024,  # size of codebook
    "weight_decay": 0.00001,  # from paper
    "lr": 0.0001,
    "epochs": 400,
    "data_root": "/ocean/projects/cis260059p/shared/AudioVGen/VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video",  # root of audio-video data
    "batch_size": 256,  # batch size for training
    "checkpoint_dir": "./checkpoints",
    "pct_start" : 0.2,
    "scheduler": True,
    "ratio":3
}

#get data dir

# Datasets
train_dataset, valid_dataset = get_datasets(config["data_root"], validation_ratio=0.004)

# Dataloaders
train_loader = DataLoader(
    dataset=train_dataset,
    num_workers=12,
    batch_size=config["batch_size"],
    pin_memory=True,
    persistent_workers=True,
    shuffle=True,
    prefetch_factor=2
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    num_workers=4,
    batch_size=config['batch_size'],
    pin_memory=True,
    prefetch_factor=2
)

# [dac, clip, s3d]
# print dataset metainfo
retlist = train_dataset.__getitem__(0)
print(f"dac dimensions {retlist[0].shape}")
print(f"clip dimensions {retlist[1].shape}")
print(f"s3d dimensions {retlist[2].shape}")

model = AudioVGen(
    config["seq_len"],
    config["embed_dim"],
    config["n_heads"],
    config["d_state"], 
    config["d_conv"], 
    config["expand"], 
    config["c_dim"],
    config["s_dim"],
    config["M"],
    config["K"],
    config["codebook_size"],
    ratio=config['ratio']
).to(DEVICE).to(torch.bfloat16)
summary(model)
model = torch.compile(model)

#setup EMA
decay = 0.999
ema_avg_fn = get_ema_multi_avg_fn(decay)
ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)

#load DAC
dac_model_path = dac.utils.download(model_type="44khz")
dac_model = dac.DAC.load(dac_model_path)
dac_model.to("cuda")
dac_model.eval()
for p in dac_model.parameters():
    p.requires_grad = False

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.AdamW(
    model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
)

# Scheduler
scheduler = None
if config['scheduler']: 
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=3000,
    )

wandb.login(key="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")

run_name = "mamba_adaln2_ablation_3"

run = wandb.init(
    name = run_name,
    entity = "AudioVGen",
    project = "AudioVGen_plots",
    config = config,
)

# Ensure checkpoint directory exists
os.makedirs(config["checkpoint_dir"], exist_ok=True)

def save_model(model, ema_model, optimizer, scheduler, metrics, epoch, path):
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
    if scheduler is not None: 
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "epoch": epoch,
            },
            path,
        )
        print(f"Checkpoint saved at {path}")
    else: 
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(), 
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "epoch": epoch,
            },
            path,
        )
        print(f"Checkpoint saved at {path}")

start_epoch = 0
best_distance = np.inf
inference_metrics = Metrics()

gc.collect()
torch.cuda.empty_cache()

#load a model for training resume
load = True
if load: 
    print('loading a model for training')
    map_location = DEVICE
    checkpoint = torch.load(
        '/ocean/projects/cis260059p/lundgren/AudioVGen/checkpoints/last_mamba_adaln2_ablation_3.pth', 
        map_location=map_location
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']  + 1
    print('model loaded')

# torch.autograd.set_detect_anomaly(True)
for epoch in range(start_epoch, config["epochs"]):
    print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

    # -----------------------------
    # Train
    # -----------------------------
    train_loss = train_epoch(
        model, ema_model, train_loader, optimizer, scheduler, DEVICE, criterion
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
    FDM, FDD, FAD = valid_epoch(
        model, 
        ema_model,
        dac_model, 
        valid_loader,
        DEVICE,
        inference_metrics,
    )
    print(f"Val (Cls) | Distance: {FDM:.4f}")
    metrics.update({
            "FDM": FDM,
    })

    # -----------------------------
    # Save checkpoints
    # -----------------------------
    checkpoint_path = os.path.join(config["checkpoint_dir"], f"last_{run_name}.pth")
    save_model(model, ema_model, optimizer, scheduler, metrics, epoch, checkpoint_path)
    print(f"Saved last epoch model: {checkpoint_path}")

    # Save model with best validation FDM distance
    if best_distance >= FDM:
        best_distance = FDM
        best_distance_path = os.path.join(config["checkpoint_dir"], "best_distance.pth")
        save_model(model, ema_model, optimizer, scheduler, metrics, epoch, best_distance_path)
        print(f"Saved best distance validation model: {best_distance_path}")

    # -----------------------------
    # Log metrics
    # -----------------------------
    if "run" in globals() and run is not None:
        run.log(metrics)

    if scheduler is not None: 
        scheduler.step()
