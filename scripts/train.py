import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json

from model.unet import UNet
from utils.data_preprocess import COCO_Dataset
from utils.loss_fn import get_loss_fn
from utils.train_utils import train, evaluate


def main():
    # Load config
    with open("config/config.json") as f:
        config = json.load(f)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    # TensorBoard Writer
    writer = SummaryWriter(
        log_dir=f"runs/unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Dataset & DataLoader
    train_dataset = COCO_Dataset(
        config["train_dir"], config["train_ann"], config["image_size"]
    )
    val_dataset = COCO_Dataset(
        config["val_dir"], config["val_ann"], config["image_size"]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    # Model, Loss, Optimizer
    model = UNet(num_classes=config["num_classes"]).to(device)
    criterion = get_loss_fn("cross_entropy")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_iou = 0.0

    for epoch in range(config["epochs"]):
        train_loss = train(model, train_loader, criterion, optimizer, device, writer=writer, epoch=epoch, total_epochs=config["epochs"])
        val_loss, val_iou = evaluate(model, val_loader, criterion, device)

        # TensorBoard 로그 기록
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        print(
            f"\n Epoch {epoch+1}/{config['epochs']} Finished | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}\n"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config["save_path"])
            print(" Best model saved.")

    writer.close()

if __name__ == "__main__":
    main()
