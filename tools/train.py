"""
EAST Training Script

Trains the EAST model on ICDAR 2015 using PyTorch.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from east.models.east_model import EAST
from east.datasets.icdar_dataset import ICDARDataset

# Simple loss: BCE for score, SmoothL1 for geometry
class EASTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
    def forward(self, pred_score, pred_geo, gt_score, gt_geo):
        loss_score = self.bce(pred_score, gt_score)
        loss_geo = self.smooth_l1(pred_geo, gt_geo)
        return loss_score + loss_geo, loss_score, loss_geo


def main():
    # Config
    images_dir = 'data/icdar2015/train/images'
    annotations_dir = 'data/icdar2015/train/annotations'
    val_split = 0.1
    batch_size = 4
    num_epochs = 10
    lr = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & loader
    dataset = ICDARDataset(images_dir, annotations_dir)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = EAST().to(device)
    # Loss & optimizer
    criterion = EASTLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch['image'].to(device)
            gt_score = batch['score_map'].to(device)
            gt_geo = batch['geometry_map'].to(device)
            optimizer.zero_grad()
            pred_score, pred_geo = model(images)
            loss, loss_score, loss_geo = criterion(pred_score, pred_geo, gt_score, gt_geo)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                gt_score = batch['score_map'].to(device)
                gt_geo = batch['geometry_map'].to(device)
                pred_score, pred_geo = model(images)
                loss, _, _ = criterion(pred_score, pred_geo, gt_score, gt_geo)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Checkpointing
        torch.save(model.state_dict(), f'checkpoints/east_last.pt')
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'checkpoints/east_best.pt')
            print("Best model updated.")

    print("Training complete!")

if __name__ == "__main__":
    main()
