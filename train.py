import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from model import UNet
from dataset import SyntheticDataset, BrainTumorDataset
from utils import save_checkpoint, check_accuracy, save_predictions_as_imgs
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128 # Increased for A100
NUM_EPOCHS = 100 # Increased epochs
NUM_WORKERS = 4
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "Dataset/train_images/"
TRAIN_MASK_DIR = "Dataset/train_masks/"
VAL_IMG_DIR = "Dataset/val_images/"
VAL_MASK_DIR = "Dataset/val_masks/"
BASE_CHANNELS = 128 # Increased model capacity

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_fn(loader, model, optimizer, loss_fn, scaler, rank):
    # Only show progress bar on rank 0
    loop = tqdm(loader) if rank == 0 else loader

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(rank)
        targets = targets.float().to(rank)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        if rank == 0:
            loop.set_postfix(loss=loss.item())

def main(rank, world_size):
    setup(rank, world_size)
    print(f"Rank {rank} initialized.")
    
    # Create model and move it to GPU with id rank
    model = UNet(n_channels=3, n_classes=1, base_channels=BASE_CHANNELS).to(rank)
    model = DDP(model, device_ids=[rank])
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Use SyntheticDataset if directories don't exist
    if not os.path.exists(TRAIN_IMG_DIR):
        if rank == 0:
            print("Dataset directories not found. Using SyntheticDataset.")
        train_ds = SyntheticDataset(num_samples=10000, img_size=(IMAGE_HEIGHT, IMAGE_WIDTH)) # Increased samples
        val_ds = SyntheticDataset(num_samples=1000, img_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    else:
        train_ds = BrainTumorDataset(
            images_dir=TRAIN_IMG_DIR,
            masks_dir=TRAIN_MASK_DIR,
        )
        val_ds = BrainTumorDataset(
            images_dir=VAL_IMG_DIR,
            masks_dir=VAL_MASK_DIR,
        )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        sampler=val_sampler,
    )

    if LOAD_MODEL:
        # load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        pass

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler, rank)

        if rank == 0:
            # save model
            checkpoint = {
                "state_dict": model.module.state_dict(), # Note: model.module for DDP
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            # check accuracy
            check_accuracy(val_loader, model.module, device=rank)

            # print some examples to a folder
            if not os.path.exists("saved_images"):
                os.makedirs("saved_images")
            save_predictions_as_imgs(
                val_loader, model.module, folder="saved_images/", device=rank
            )
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs. Spawning processes...")
    # For 4x A100, world_size should be 4
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
