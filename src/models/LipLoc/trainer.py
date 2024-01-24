import argparse
import itertools

import numpy as np
import torch
from tqdm.autonotebook import tqdm
import importlib
import wandb
import os

from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass
import tyro


### Global Stuff ####
@dataclass
class Args:
    expid: str = 'exp_default'

args = tyro.cli(Args)

CFG = importlib.import_module(f"config.{args.expid}").CFG
model = importlib.import_module(f"models.{CFG.model}").Model(CFG)
get_dataloader = importlib.import_module(f"dataloaders.{CFG.dataloader}").get_dataloader
get_filenames = importlib.import_module(f"dataloaders.{CFG.dataloader}").get_filenames

wandb.init(
    project="cross-modal-place-recognition",
    mode='disabled',
    # track hyperparameters and run metadata
    config={
            "image_model_name": "resnet50",
        "epochs": 100,
    }
)
####################


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def make_train_valid_dfs():
        image_ids = get_filenames(CFG.train_sequences, CFG.data_path, CFG.data_path_360)
        np.random.seed(42)
        valid_ids = np.random.choice(
                    image_ids, size=int(0.2 * len(image_ids)), replace=False
        )
        train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
        return train_ids, valid_ids



def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["camera_image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(
            train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items()}
        loss = model(batch)

        count = batch["camera_image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    print(CFG.details)
    dirs_to_create = [CFG.expdir, CFG.logdir]
    for dirs in dirs_to_create:
        os.makedirs(dirs, exist_ok=True)

    train_df, valid_df = make_train_valid_dfs()
    print(f"Train: {len(train_df)} Valid: {len(valid_df)}")

    train_loader = get_dataloader(train_df, mode="train", CFG=CFG)
    valid_loader = get_dataloader(valid_df, mode="valid", CFG=CFG)

    model.to(CFG.device)
    params = [
        {
            "params": model.encoder_camera.parameters(), 
            "lr": CFG.text_encoder_lr
        },
        {
            "params": model.encoder_lidar.parameters(),
            "lr": CFG.image_encoder_lr
        },
        {
            "params": itertools.chain(model.projection_lidar.parameters(), model.projection_camera.parameters()), 
            "lr": CFG.head_lr, 
            "weight_decay": CFG.weight_decay
        }
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    writer = SummaryWriter(log_dir=CFG.logdir)

    wandb.watch(model)
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(
            model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)

        wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
        writer.add_scalar("train_loss", train_loss.avg, epoch)
        writer.add_scalar("valid_loss", valid_loss.avg, epoch)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), CFG.best_model_path)
            print("Saved Best Model!")
        torch.save(model.state_dict(), CFG.final_model_path)
        lr_scheduler.step(valid_loss.avg)

    wandb.finish()


if __name__ == "__main__":
    main()
