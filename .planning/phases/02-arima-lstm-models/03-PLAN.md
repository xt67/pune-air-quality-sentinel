---
phase: 2
plan: 3
title: "LSTM with Attention"
wave: 2
depends_on: [1]
files_modified:
  - src/models/lstm.py
  - tests/test_lstm.py
requirements_addressed: [LSTM-02, LSTM-03]
autonomous: true
---

# Plan 3: LSTM with Attention

<objective>
Implement stacked LSTM architecture with temporal attention mechanism for AQI forecasting. Optimized for RTX 4050 (6GB VRAM).
</objective>

<must_haves>
- Stacked LSTM: 128 → 64 hidden units
- Temporal attention mechanism
- HuberLoss for robustness to outliers
- Support for multi-step forecasting (horizon=24/48)
- Mixed precision compatible
- Unit tests
</must_haves>

## Tasks

<task id="3.1">
<title>Create LSTM architecture</title>
<action>
Create src/models/lstm.py:

```python
"""LSTM with attention for AQI forecasting."""
import torch
import torch.nn as nn
from typing import Tuple


class TemporalAttention(nn.Module):
    """Attention mechanism over temporal dimension."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_size)
            
        Returns:
            context: (batch, hidden_size)
            weights: (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)
        
        # Weighted sum of LSTM outputs
        context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)
        
        return context, weights


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM with attention for AQI forecasting.
    
    Architecture:
    - Input projection
    - LSTM layer 1 (128 units)
    - LSTM layer 2 (64 units)
    - Temporal attention
    - Output projection to horizon
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, int] = (128, 64),
        horizon: int = 24,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.horizon = horizon
        h1, h2 = hidden_sizes
        
        # Input projection
        self.input_proj = nn.Linear(input_size, h1)
        
        # Stacked LSTM
        self.lstm1 = nn.LSTM(h1, h1, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True, dropout=dropout)
        
        # Attention
        self.attention = TemporalAttention(h2)
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(h2, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, horizon)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            
        Returns:
            predictions: (batch, horizon)
        """
        # Project input
        x = self.input_proj(x)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Attention
        context, _ = self.attention(x)
        
        # Output
        return self.output(context)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(
    input_size: int,
    horizon: int = 24,
    device: str = "cuda"
) -> LSTMForecaster:
    """Create and initialize LSTM model."""
    model = LSTMForecaster(
        input_size=input_size,
        hidden_sizes=(128, 64),
        horizon=horizon,
        dropout=0.2
    )
    
    # Initialize weights
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
            
    return model.to(device)
```
</action>
</task>

<task id="3.2">
<title>Create unit tests</title>
<action>
Create tests/test_lstm.py with tests for:
- Model initialization
- Forward pass shape
- Attention mechanism
- Parameter count
- GPU compatibility
</action>
</task>

<acceptance_criteria>
- `python -c "from src.models.lstm import LSTMForecaster; print('OK')"` succeeds
- `pytest tests/test_lstm.py -v` all pass
- Forward pass returns (batch, horizon) shape
- Model fits in 6GB VRAM
</acceptance_criteria>
