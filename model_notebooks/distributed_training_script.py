import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
import gc
import os
import numpy as np

from models import MaskVatAdaLN
from datasets import get_datasets
from training import train_epoch, valid_epoch
from datasets import Metrics
from torchinfo import summary

#distributed imports
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

####### DISTRIBUTD TRAINING INITIALIZATION #############

def distributed_setup(): 
    dist.init_process_group(backend="nccl")
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
    "epochs": 50,
    "data_root": "./VGGSound_raw_data/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video",  # root of audio-video data
    "batch_size": 16,  # batch size for training
    "checkpoint_dir": "./checkpoints",
}

####### DATASETS AND DATA LOADERS #########

train_dataset, valid_dataset = get_datasets(config["data_root"])

train_sampler = DistributedSampler(train_dataset)
val_sampler = DistributedSampler(valid_dataset)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config["batch_size"],
    sampler=train_sampler, 
    shuffle=False
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=config["batch_size"],
    sampler=val_sampler,
    shuffle=False
)

# [dac, clip, s3d]
# print dataset metainfo
if local_rank == 0: 
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
).to(local_rank)
model = DDP(model, device_ids=[local_rank])

if local_rank == 0: 
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

if local_rank == 0: 
    wandb.login(key="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")

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
            "model_state_dict": model.module.state_dict(),
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

inference_metrics = Metrics()

gc.collect()
torch.cuda.empty_cache()

# torch.autograd.set_detect_anomaly(True)
for epoch in range(start_epoch, config["epochs"]):
    if local_rank == 0: 
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")

    #set sampler epoch
    train_loader.sampler.set_epoch(epoch)
    valid_loader.sampler.set_epoch(epoch)

    # -----------------------------
    # Train
    # -----------------------------
    train_loss = train_epoch(
        model, train_loader, optimizer, 
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
    # -----------------------------
    FDM, FDD, FAD = valid_epoch(
        model,
        valid_loader,
        DEVICE,
        inference_metrics,
        distributed=True, 
        rank=local_rank
    )

    if local_rank == 0: 
        print(f"Val (Cls) | Distance: {FDM:.4f}")
    metrics['FDM'] = FDM

    # -----------------------------
    # Save checkpoints
    # -----------------------------
    if local_rank == 0: 
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"last_{run_name}.pth")
        save_model(model, optimizer, scheduler, metrics, epoch, checkpoint_path)
        print(f"Saved last epoch model: {checkpoint_path}")

        # Save model with best validation FDM distance
        if best_distance >= FDM:
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

dist.destroy_process_group()
