import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import gc

import os

from models import MaskVatAdaLN
from datasets import get_datasets
from testing import valid_epoch
from datasets import Metrics

from torchinfo import summary
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import dac

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", DEVICE)

config = {
    "seq_len": 862,  # seq_len of DAC encoding
    "embed_dim": 1024,  # embed_dim for the model
    "n_heads": 16,  # number of heads for multiheaded attention
    "c_dim": 512,  # dimensions of clip encoding
    "s_dim": 1024,  # dimensions of s3d encoding
    "M": 24,  # number of AdaLN Blocks in the model
    "K": 9,  # dimensions of DAC encoding
    "codebook_size": 1024,  # size of codebook
    "weight_decay": 0.00001,  # from paper
    "lr": 0.0001,
    "epochs": 400,
    "data_root": "/ocean/projects/cis260059p/shared/AudioVGen/VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video",  # root of audio-video data
    "batch_size": 128,  # batch size for training
    "checkpoint_dir": "./checkpoints",
    "pct_start": 0.2
}

# Datasets
_, valid_dataset = get_datasets(config["data_root"], validation_ratio=0.004)

# Dataloaders
valid_loader = DataLoader(
    dataset=valid_dataset,
    num_workers=8,
    batch_size=config["batch_size"],
    pin_memory=True,
    prefetch_factor=4
)

# [dac, clip, s3d]
# print dataset metainfo
retlist = valid_dataset.__getitem__(0)
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
scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,
    total_iters=20,
)

# Mixed-Precision Training
scaler = torch.amp.GradScaler(device="cuda")

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
best_distance = 5.73
inference_metrics = Metrics(True)

gc.collect()
torch.cuda.empty_cache()

#load a model for training resume
load = True
if load: 
    print('loading a model for training')
    map_location = DEVICE
    checkpoint = torch.load(
        '/ocean/projects/cis260059p/adai1/AudioVGen/checkpoints/last_big_model_run.pth', 
        map_location=map_location
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']  + 1
    print('model loaded')

# make dir for saving videos
os.makedirs("./videos", exist_ok=True)

# obtain and print metrics
FDM, FDD, FAD = valid_epoch(
    model, 
    ema_model,
    dac_model, 
    valid_loader,
    DEVICE,
    inference_metrics,
    steps=32,
)

print("FDM:", FDM)
print("FDD:", FDD)
print("FAD:", FAD)
