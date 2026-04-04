"""
Spatio-Temporal Graph Neural Network for multi-node AQI forecasting.

Combines GRU for temporal patterns with GCN for spatial dependencies.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNConv(nn.Module):
    """
    Simple Graph Convolutional layer (without torch_geometric dependency).
    
    Implements: H' = A_norm @ H @ W + b
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        # x: (B, N, F_in) -> (B, N, F_out)
        support = torch.matmul(x, self.weight)  # (B, N, F_out)
        output = torch.matmul(adj, support)      # (B, N, F_out)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class TemporalBlock(nn.Module):
    """
    Temporal processing block using GRU.
    
    Captures temporal patterns across the sequence dimension.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            h0: Initial hidden state
            
        Returns:
            output: GRU outputs (batch, seq_len, hidden_dim)
            hn: Final hidden state
        """
        output, hn = self.gru(x, h0)
        output = self.dropout(output)
        return output, hn


class SpatialBlock(nn.Module):
    """
    Spatial processing block using GCN.
    
    Captures spatial dependencies between nodes using graph convolution.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gcn1 = SimpleGCNConv(input_dim, hidden_dim)
        self.gcn2 = SimpleGCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, num_nodes, input_dim)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch, num_nodes, output_dim)
        """
        h = self.gcn1(x, adj)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.gcn2(h, adj)
        h = self.norm(h)
        return h


class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network.
    
    Architecture:
    1. Temporal Block: GRU processes each node's time series
    2. Spatial Block: GCN aggregates information across nodes
    3. Output Layer: Predicts future AQI for each node
    
    Args:
        num_nodes: Number of spatial nodes (stations)
        input_dim: Number of input features per node per timestep
        hidden_dim: Hidden dimension for GRU and GCN
        output_dim: Output dimension (forecast horizon)
        gru_layers: Number of GRU layers
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 24,
        gru_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal processing (shared across nodes)
        self.temporal = TemporalBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=gru_layers,
            dropout=dropout,
        )
        
        # Spatial processing
        self.spatial = SpatialBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Residual connection for temporal
        self.temporal_residual = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, num_nodes, input_dim)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Predictions (batch, num_nodes, output_dim)
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # Ensure adj is on same device
        if adj.device != x.device:
            adj = adj.to(x.device)
        
        # Project input features
        # (B, T, N, F) -> (B, T, N, H)
        x = self.input_proj(x)
        
        # Temporal processing for each node
        # Reshape to process all nodes: (B*N, T, H)
        x_temp = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, -1)
        temporal_out, _ = self.temporal(x_temp)
        
        # Take last timestep output: (B*N, H)
        temporal_out = temporal_out[:, -1, :]
        
        # Reshape back: (B, N, H)
        temporal_out = temporal_out.reshape(batch_size, num_nodes, -1)
        
        # Residual from last input timestep
        residual = self.temporal_residual(x[:, -1, :, :])  # (B, N, H)
        temporal_out = temporal_out + residual
        
        # Spatial processing: (B, N, H) -> (B, N, H)
        spatial_out = self.spatial(temporal_out, adj)
        
        # Residual connection
        spatial_out = spatial_out + temporal_out
        
        # Output projection: (B, N, H) -> (B, N, output_dim)
        output = self.output_proj(spatial_out)
        
        return output
    
    def predict(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make predictions (inference mode).
        
        Args:
            x: Input tensor (batch, seq_len, num_nodes, input_dim)
            adj: Normalized adjacency matrix
            
        Returns:
            Predictions (batch, num_nodes, output_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x, adj)


class STGNNTrainer:
    """Trainer for ST-GNN model."""
    
    def __init__(
        self,
        model: STGNN,
        adj: torch.Tensor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.adj = adj.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )
        self.criterion = nn.HuberLoss(delta=1.0)
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x, self.adj)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x, self.adj)
                loss = self.criterion(predictions, batch_y)
                mae = torch.mean(torch.abs(predictions - batch_y))
                
                total_loss += loss.item()
                total_mae += mae.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        
        self.scheduler.step(avg_loss)
        
        return avg_loss, avg_mae
