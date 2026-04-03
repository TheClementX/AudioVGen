import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
import gc
import os
import numpy as np
from configparser import ConfigParser

from models import MaskVatAdaLN
from datasets import get_datasets
from training import train_epoch, valid_epoch
from datasets import Metrics
from torchinfo import summary
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import dac

#distributed imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

####### DISTRIBUTD TRAINING INITIALIZATION #############

def distributed_setup(): 
    #added timeout increase for validation epoch
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    #return local rank (ie local gpu number)
    return local_rank

local_rank = distributed_setup()

DEVICE =  torch.cuda.current_device()
if local_rank == 0: 
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
    "epochs": 400,
    "data_root": "./VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video",  # root of audio-video data
    "batch_size": 256,  # batch size for training
    "checkpoint_dir": "./checkpoints",
    "pct_start": 0.2
}

####### DATASETS AND DATA LOADERS #########

train_dataset, valid_dataset = get_datasets(config["data_root"])

train_sampler = DistributedSampler(train_dataset)

train_loader = DataLoader(
    dataset=train_dataset,
    num_workers=12,
    batch_size=config["batch_size"],
    pin_memory=True,
    prefetch_factor=4,
    sampler=train_sampler, 
    shuffle=False
)

#no sampler for single gpu inference 
valid_loader = DataLoader(
    dataset=valid_dataset,
    num_workers=8,
    prefetch_factor=4,
    batch_size=config["batch_size"],
    pin_memory=True, 
    shuffle=False
)

# [dac, clip, s3d]
# print dataset metainfo
if local_rank == 0: 
    retlist = train_dataset.__getitem__(0)
    print(f"dac dimensions {retlist[0].shape}")
    print(f"clip dimensions {retlist[1].shape}")
    print(f"s3d dimensions {retlist[2].shape}")

#init model
model = MaskVatAdaLN(
    config["seq_len"],
    config["embed_dim"],
    config["n_heads"],
    config["c_dim"],
    config["s_dim"],
    config["M"],
    config["K"],
    config["codebook_size"],
).to(local_rank)

#distribute model
model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

#create ema model
decay = 0.999
ema_avg_fn = get_ema_multi_avg_fn(decay)
ema_model = AveragedModel(model.module, multi_avg_fn=ema_avg_fn)

#load DAC for decoding
dac_model_path = dac.utils.download(model_type="44khz")
dac_model = dac.DAC.load(dac_model_path)
dac_model.to("cuda")
dac_model.eval()
for p in dac_model.parameters():
    p.requires_grad = False

#print model architecture
if local_rank == 0: 
    summary(model)

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

if local_rank == 0:
    CF = ConfigParser()
    CF.read("./config.ini")
    wandb_key = CF.get("Wandb", "key")
    wandb.login(key='')

    run_name = "distributed_run_1"

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
    torch.save(
        {
            "model_state_dict": model.module.state_dict(),
            "ema_model_state_dict": ema_model.state_dict(), 
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
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

load = True
if load: 
    print('loading a model for training')
    map_location = DEVICE
    checkpoint = torch.load(
        '/ocean/projects/cis260059p/shared/AudioVGen/checkpoints/last_single_trining_run_const_lr.pth', 
        map_location=f"cuda:{map_location}"
    )
    model.module.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print('model loaded')

# torch.autograd.set_detect_anomaly(True)
for epoch in range(start_epoch, config["epochs"]):
    if local_rank == 0: 
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

    #set sampler epoch
    train_loader.sampler.set_epoch(epoch)

    # -----------------------------
    # Train
    # -----------------------------
    train_loss = train_epoch(
        model, ema_model, train_loader, optimizer, 
        scheduler, scaler, DEVICE, criterion,
        distributed=True, 
        rank=local_rank
    )

    curr_lr = optimizer.param_groups[0]["lr"]
    if local_rank == 0: 
        print(f"Train | Loss: {train_loss:.4f} | LR: {curr_lr:.6f}")

    metrics = {
        "train_loss": train_loss,
        "lr": curr_lr,
    }

    # -----------------------------
    # Validation and compute frechet distance
    # run validation and checkpointing on only a single gpu
    # -----------------------------
    if local_rank == 0: 
        FDM, FDD, FAD = valid_epoch(
            model,
            ema_model,
            dac_model, 
            valid_loader,
            DEVICE,
            inference_metrics,
            distributed=False, 
        )

        print(f"Val (Cls) | Distance: {FDM:.4f}")
        metrics['FDM'] = FDM

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

    dist.barrier()

dist.destroy_process_group()
