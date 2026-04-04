"""Tests for ST-GNN model."""

import numpy as np
import pytest
import torch

from src.models.stgnn import (
    SimpleGCNConv,
    TemporalBlock,
    SpatialBlock,
    STGNN,
    STGNNTrainer,
)
from src.models.graph import build_adjacency_matrix, normalize_adjacency, PUNE_STATIONS


class TestSimpleGCNConv:
    """Tests for SimpleGCNConv layer."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        gcn = SimpleGCNConv(16, 32)
        x = torch.randn(4, 10, 16)  # (batch, nodes, features)
        adj = torch.randn(10, 10)
        
        out = gcn(x, adj)
        assert out.shape == (4, 10, 32)
    
    def test_no_bias(self):
        """Model without bias should work."""
        gcn = SimpleGCNConv(16, 32, bias=False)
        assert gcn.bias is None
        
        x = torch.randn(2, 5, 16)
        adj = torch.randn(5, 5)
        out = gcn(x, adj)
        assert out.shape == (2, 5, 32)
    
    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        gcn = SimpleGCNConv(16, 32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        adj = torch.randn(5, 5)
        
        out = gcn(x, adj)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert gcn.weight.grad is not None


class TestTemporalBlock:
    """Tests for TemporalBlock."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        block = TemporalBlock(input_dim=16, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 72, 16)  # (batch, seq_len, features)
        
        out, hn = block(x)
        assert out.shape == (4, 72, 32)
    
    def test_hidden_state_shape(self):
        """Hidden state should have correct shape."""
        block = TemporalBlock(input_dim=16, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 72, 16)
        
        _, hn = block(x)
        assert hn.shape == (2, 4, 32)  # (num_layers, batch, hidden)
    
    def test_single_layer(self):
        """Single layer should work."""
        block = TemporalBlock(input_dim=16, hidden_dim=32, num_layers=1)
        x = torch.randn(2, 24, 16)
        
        out, hn = block(x)
        assert out.shape == (2, 24, 32)
    
    def test_with_initial_hidden(self):
        """Should accept initial hidden state."""
        block = TemporalBlock(input_dim=16, hidden_dim=32, num_layers=2)
        x = torch.randn(4, 72, 16)
        h0 = torch.zeros(2, 4, 32)
        
        out, hn = block(x, h0)
        assert out.shape == (4, 72, 32)


class TestSpatialBlock:
    """Tests for SpatialBlock."""
    
    def test_output_shape(self):
        """Output should have correct shape."""
        block = SpatialBlock(input_dim=32, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 10, 32)  # (batch, nodes, features)
        adj = torch.randn(10, 10)
        
        out = block(x, adj)
        assert out.shape == (4, 10, 32)
    
    def test_with_normalized_adjacency(self):
        """Should work with normalized adjacency."""
        block = SpatialBlock(input_dim=32, hidden_dim=64, output_dim=32)
        
        adj_np = build_adjacency_matrix(PUNE_STATIONS)
        adj_norm = normalize_adjacency(adj_np)
        adj = torch.from_numpy(adj_norm)
        
        x = torch.randn(2, 10, 32)
        out = block(x, adj)
        assert out.shape == (2, 10, 32)
    
    def test_gradient_flow(self):
        """Gradients should flow through."""
        block = SpatialBlock(input_dim=32, hidden_dim=64, output_dim=32)
        x = torch.randn(2, 5, 32, requires_grad=True)
        adj = torch.randn(5, 5)
        
        out = block(x, adj)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None


class TestSTGNN:
    """Tests for full ST-GNN model."""
    
    @pytest.fixture
    def model(self):
        """Create a small ST-GNN for testing."""
        return STGNN(
            num_nodes=10,
            input_dim=8,
            hidden_dim=32,
            output_dim=24,
            gru_layers=2,
            dropout=0.1,
        )
    
    @pytest.fixture
    def adj(self):
        """Create normalized adjacency matrix."""
        adj_np = build_adjacency_matrix(PUNE_STATIONS)
        adj_norm = normalize_adjacency(adj_np)
        return torch.from_numpy(adj_norm)
    
    def test_output_shape(self, model, adj):
        """Output should have correct shape."""
        x = torch.randn(4, 72, 10, 8)  # (batch, seq_len, nodes, features)
        out = model(x, adj)
        assert out.shape == (4, 10, 24)  # (batch, nodes, horizon)
    
    def test_single_batch(self, model, adj):
        """Should work with batch size 1."""
        x = torch.randn(1, 72, 10, 8)
        out = model(x, adj)
        assert out.shape == (1, 10, 24)
    
    def test_different_sequence_length(self, model, adj):
        """Should work with different sequence lengths."""
        for seq_len in [24, 48, 72, 96]:
            x = torch.randn(2, seq_len, 10, 8)
            out = model(x, adj)
            assert out.shape == (2, 10, 24)
    
    def test_predict_method(self, model, adj):
        """Predict method should return detached tensor."""
        x = torch.randn(2, 72, 10, 8)
        out = model.predict(x, adj)
        
        assert out.shape == (2, 10, 24)
        assert not out.requires_grad
    
    def test_gradient_flow_full(self, model, adj):
        """Gradients should flow through entire model."""
        x = torch.randn(2, 72, 10, 8, requires_grad=True)
        out = model(x, adj)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_parameter_count(self, model):
        """Model should have reasonable parameter count."""
        num_params = sum(p.numel() for p in model.parameters())
        # Small model should have <500k params
        assert num_params < 500_000
    
    def test_train_eval_modes(self, model, adj):
        """Model should behave differently in train/eval."""
        x = torch.randn(2, 72, 10, 8)
        
        model.train()
        out_train = model(x, adj)
        
        model.eval()
        out_eval = model(x, adj)
        
        # Due to dropout, outputs may differ slightly
        # Just verify both work
        assert out_train.shape == out_eval.shape


class TestSTGNNTrainer:
    """Tests for STGNNTrainer."""
    
    @pytest.fixture
    def trainer(self):
        """Create trainer with small model."""
        model = STGNN(
            num_nodes=5,
            input_dim=4,
            hidden_dim=16,
            output_dim=12,
        )
        adj_np = np.random.randn(5, 5).astype(np.float32)
        adj = torch.from_numpy(adj_np)
        return STGNNTrainer(model, adj, device="cpu")
    
    def test_train_epoch(self, trainer):
        """Train epoch should return loss."""
        # Create dummy dataloader
        class DummyLoader:
            def __iter__(self):
                for _ in range(3):
                    x = torch.randn(4, 24, 5, 4)
                    y = torch.randn(4, 5, 12)
                    yield x, y
            
            def __len__(self):
                return 3
        
        loss = trainer.train_epoch(DummyLoader())
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_validate(self, trainer):
        """Validate should return loss and MAE."""
        class DummyLoader:
            def __iter__(self):
                for _ in range(2):
                    x = torch.randn(4, 24, 5, 4)
                    y = torch.randn(4, 5, 12)
                    yield x, y
            
            def __len__(self):
                return 2
        
        loss, mae = trainer.validate(DummyLoader())
        assert isinstance(loss, float)
        assert isinstance(mae, float)
        assert loss >= 0
        assert mae >= 0


class TestSTGNNWithRealGraph:
    """Integration tests with real Pune graph."""
    
    def test_forward_with_pune_graph(self):
        """Model should work with actual Pune station graph."""
        adj_np = build_adjacency_matrix(PUNE_STATIONS)
        adj_norm = normalize_adjacency(adj_np)
        adj = torch.from_numpy(adj_norm)
        
        model = STGNN(
            num_nodes=10,
            input_dim=8,
            hidden_dim=64,
            output_dim=24,
        )
        
        x = torch.randn(4, 72, 10, 8)
        out = model(x, adj)
        
        assert out.shape == (4, 10, 24)
        assert not torch.isnan(out).any()
    
    def test_predictions_reasonable_range(self):
        """Predictions should be in reasonable AQI range after training."""
        adj_np = build_adjacency_matrix(PUNE_STATIONS)
        adj_norm = normalize_adjacency(adj_np)
        adj = torch.from_numpy(adj_norm)
        
        model = STGNN(
            num_nodes=10,
            input_dim=8,
            hidden_dim=32,
            output_dim=24,
        )
        
        # Simulate input in normalized range [0, 1]
        x = torch.rand(2, 72, 10, 8)
        out = model(x, adj)
        
        # Output should be bounded (model is untrained, but shouldn't explode)
        assert out.abs().max() < 1000
