---
phase: 2
plan: 4
title: "Training Script with Early Stopping"
wave: 3
depends_on: [1, 3]
files_modified:
  - src/models/train.py
requirements_addressed: [LSTM-04, LSTM-05, LSTM-06, LSTM-07]
autonomous: true
---

# Plan 4: Training Script with Early Stopping

<objective>
Create training script with early stopping, mixed precision training, and checkpoint management. Optimized to complete in ≤30 minutes on RTX 4050.
</objective>

<must_haves>
- HuberLoss for robustness
- Adam optimizer with lr=1e-3
- Early stopping (patience=10 on val MAE)
- Mixed precision (AMP) for VRAM efficiency
- Best model checkpointing by val MAE
- TensorBoard logging (optional)
- Progress bar with metrics
</must_haves>

## Tasks

<task id="4.1">
<title>Create training script</title>
<action>
Create src/models/train.py:

```python
"""Training script for LSTM forecaster."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop


class Trainer:
    """LSTM trainer with mixed precision and early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-3,
        device: str = "cuda",
        checkpoint_dir: str = "models",
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss and optimizer
        self.criterion = nn.HuberLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Mixed precision
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.history = {"train_loss": [], "val_loss": [], "val_mae": []}
        self.best_val_mae = float("inf")
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate and return loss and MAE."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_targets = [], []
        
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            
            if self.use_amp:
                with autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
                
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            
        val_loss = total_loss / len(self.val_loader)
        
        # Calculate MAE
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        mae = torch.mean(torch.abs(preds - targets)).item()
        
        return val_loss, mae
    
    def save_checkpoint(self, path: str, epoch: int, val_mae: float):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_mae": val_mae,
            "history": self.history
        }, path)
        
    def train(
        self,
        epochs: int = 100,
        patience: int = 10,
        seed: int = 42
    ) -> Dict:
        """
        Full training loop.
        
        Returns:
            Training history dict
        """
        set_seed(seed)
        early_stopping = EarlyStopping(patience=patience)
        
        pbar = tqdm(range(epochs), desc="Training")
        
        for epoch in pbar:
            train_loss = self.train_epoch()
            val_loss, val_mae = self.validate()
            
            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            
            # Update progress bar
            pbar.set_postfix({
                "train": f"{train_loss:.4f}",
                "val": f"{val_loss:.4f}",
                "mae": f"{val_mae:.2f}"
            })
            
            # Save best model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                best_path = self.checkpoint_dir / "lstm_best.pth"
                self.save_checkpoint(str(best_path), epoch, val_mae)
                logger.info(f"New best model: MAE={val_mae:.2f}")
            
            # Early stopping
            if early_stopping(val_mae):
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        # Check for NaN in gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                logger.error(f"NaN gradient in {name}")
                
        return self.history


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    horizon: int = 24,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    checkpoint_dir: str = "models",
    seed: int = 42
) -> Tuple[nn.Module, Dict]:
    """
    Train LSTM model.
    
    Returns:
        Tuple of (trained model, history dict)
    """
    from src.models.lstm import get_model
    
    model = get_model(input_size, horizon, device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Device: {device}, AMP: {device == 'cuda'}")
    
    trainer = Trainer(
        model, train_loader, val_loader,
        lr=lr, device=device, checkpoint_dir=checkpoint_dir
    )
    
    history = trainer.train(epochs=epochs, patience=patience, seed=seed)
    
    # Load best model
    best_path = Path(checkpoint_dir) / "lstm_best.pth"
    if best_path.exists():
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model (MAE={checkpoint['val_mae']:.2f})")
        
    return model, history
```
</action>
</task>

<acceptance_criteria>
- Training completes without OOM on RTX 4050
- No NaN gradients
- Best checkpoint saved at models/lstm_best.pth
- Training ≤30 minutes for 100 epochs
</acceptance_criteria>
