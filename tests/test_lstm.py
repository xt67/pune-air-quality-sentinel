"""Tests for LSTM forecaster model."""
import pytest
import torch

from src.models.lstm import LSTMForecaster, TemporalAttention, count_parameters, get_model


class TestTemporalAttention:
    """Tests for TemporalAttention module."""
    
    def test_init(self):
        """Test attention initialization."""
        attn = TemporalAttention(hidden_size=64)
        
        assert attn is not None
        
    def test_forward_shapes(self):
        """Test forward pass returns correct shapes."""
        batch_size = 8
        seq_len = 24
        hidden_size = 64
        
        attn = TemporalAttention(hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        context, weights = attn(x)
        
        assert context.shape == (batch_size, hidden_size)
        assert weights.shape == (batch_size, seq_len)
        
    def test_weights_sum_to_one(self):
        """Test attention weights sum to 1."""
        attn = TemporalAttention(hidden_size=64)
        x = torch.randn(8, 24, 64)
        
        _, weights = attn(x)
        
        sums = weights.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(8), rtol=1e-5, atol=1e-5)


class TestLSTMForecaster:
    """Tests for LSTMForecaster model."""
    
    def test_init(self):
        """Test model initialization."""
        model = LSTMForecaster(input_size=10, horizon=24)
        
        assert model.horizon == 24
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = LSTMForecaster(
            input_size=15,
            hidden_sizes=(256, 128),
            horizon=48,
            dropout=0.3
        )
        
        assert model.horizon == 48
        
    def test_forward_shape(self):
        """Test forward pass returns correct shape."""
        batch_size = 16
        seq_len = 72
        input_size = 10
        horizon = 24
        
        model = LSTMForecaster(input_size=input_size, horizon=horizon)
        x = torch.randn(batch_size, seq_len, input_size)
        
        out = model(x)
        
        assert out.shape == (batch_size, horizon)
        
    def test_forward_different_batch_sizes(self):
        """Test forward works with different batch sizes."""
        model = LSTMForecaster(input_size=10, horizon=24)
        
        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 72, 10)
            out = model(x)
            assert out.shape == (batch_size, 24)
            
    def test_forward_different_seq_lengths(self):
        """Test forward works with different sequence lengths."""
        model = LSTMForecaster(input_size=10, horizon=24)
        
        for seq_len in [24, 48, 72, 168]:
            x = torch.randn(8, seq_len, 10)
            out = model(x)
            assert out.shape == (8, 24)
            
    def test_forward_gradient_flow(self):
        """Test gradients flow through model."""
        model = LSTMForecaster(input_size=10, horizon=24)
        x = torch.randn(8, 72, 10, requires_grad=True)
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestCountParameters:
    """Tests for count_parameters utility."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = LSTMForecaster(input_size=10, horizon=24)
        
        count = count_parameters(model)
        
        assert count > 0
        assert isinstance(count, int)
        
    def test_larger_model_more_params(self):
        """Test larger model has more parameters."""
        small = LSTMForecaster(input_size=10, hidden_sizes=(64, 32), horizon=24)
        large = LSTMForecaster(input_size=10, hidden_sizes=(256, 128), horizon=24)
        
        assert count_parameters(large) > count_parameters(small)


class TestGetModel:
    """Tests for get_model factory function."""
    
    def test_get_model_cpu(self):
        """Test model creation on CPU."""
        model = get_model(input_size=10, horizon=24, device="cpu")
        
        assert model is not None
        assert next(model.parameters()).device.type == "cpu"
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_model_cuda(self):
        """Test model creation on CUDA."""
        model = get_model(input_size=10, horizon=24, device="cuda")
        
        assert next(model.parameters()).device.type == "cuda"
        
    def test_weights_initialized(self):
        """Test weights are properly initialized."""
        model = get_model(input_size=10, horizon=24, device="cpu")
        
        # Check biases are zeros
        for name, param in model.named_parameters():
            if "bias" in name:
                assert torch.allclose(param, torch.zeros_like(param))


class TestModelMemory:
    """Tests for model memory usage."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fits_in_vram(self):
        """Test model fits in 6GB VRAM with batch training."""
        model = get_model(input_size=15, horizon=24, device="cuda")
        
        # Simulate training batch
        batch_size = 64
        seq_len = 72
        x = torch.randn(batch_size, seq_len, 15, device="cuda")
        
        # Forward pass
        out = model(x)
        
        # Check VRAM usage (should be well under 6GB)
        memory_used = torch.cuda.memory_allocated() / 1e9
        
        assert memory_used < 2.0  # Should use less than 2GB for this
        
        # Clean up
        del model, x, out
        torch.cuda.empty_cache()
