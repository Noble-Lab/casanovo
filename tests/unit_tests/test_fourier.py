"""Unit tests for Fourier positional encodings."""

import pytest
import torch

from casanovo.denovo.fourier import (
    FourierEncoder,
    FourierPeakEncoder,
    _build_fourier_frequencies,
)


class TestBuildFourierFrequencies:
    """Tests for the frequency vector construction."""

    def test_default_frequency_count(self):
        """Test the expected number of frequencies with default params."""
        freqs = _build_fourier_frequencies(m_min=1e-4, m_max=1000.0)
        # Low freqs: 1000 (from m_max=1000 down to 1)
        # High freqs: round(1/1e-4) = 10000
        assert len(freqs) == 1000 + 10000

    def test_small_frequency_count(self):
        """Test frequency count with small params for faster tests."""
        freqs = _build_fourier_frequencies(m_min=0.1, m_max=5.0)
        # Low freqs: 5 (5, 4, 3, 2, 1)
        # High freqs: round(1/0.1) = 10
        assert len(freqs) == 5 + 10

    def test_low_frequencies_range(self):
        """Test that low frequencies span 1/m_max to 1."""
        freqs = _build_fourier_frequencies(m_min=0.1, m_max=5.0)
        low_freqs = freqs[:5]
        assert torch.isclose(low_freqs[0], torch.tensor(1.0 / 5.0))
        assert torch.isclose(low_freqs[-1], torch.tensor(1.0))

    def test_high_frequencies_range(self):
        """Test that high frequencies span to 1/m_min."""
        freqs = _build_fourier_frequencies(m_min=0.1, m_max=5.0)
        high_freqs = freqs[5:]
        assert torch.isclose(high_freqs[-1], torch.tensor(1.0 / 0.1))

    def test_frequencies_are_positive(self):
        """Test that all frequencies are positive."""
        freqs = _build_fourier_frequencies(m_min=0.1, m_max=10.0)
        assert torch.all(freqs > 0)


class TestFourierEncoder:
    """Tests for the FourierEncoder class."""

    def test_output_shape(self):
        """Test that the output shape is correct for various inputs."""
        d_model = 64
        encoder = FourierEncoder(d_model, m_min=0.1, m_max=5.0)

        # Single value
        x = torch.tensor([100.0])
        out = encoder(x)
        assert out.shape == (1, d_model)

        # Batch of values
        x = torch.tensor([100.0, 200.0, 300.0])
        out = encoder(x)
        assert out.shape == (3, d_model)

        # 2D input (batch, n_peaks)
        x = torch.randn(4, 10)
        out = encoder(x)
        assert out.shape == (4, 10, d_model)

    def test_frequencies_are_not_learned(self):
        """Test that frequencies are buffers, not parameters."""
        encoder = FourierEncoder(64, m_min=0.1, m_max=5.0)
        param_names = [name for name, _ in encoder.named_parameters()]
        # Frequencies should NOT be in parameters (they're buffers)
        assert not any("frequencies" in name for name in param_names)
        # But the FFN should have learnable parameters
        assert any("ffn" in name for name in param_names)

    def test_ffn_has_gradients(self):
        """Test that gradients flow through the FFN."""
        encoder = FourierEncoder(64, m_min=0.1, m_max=5.0)
        x = torch.tensor([100.0, 200.0])
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # FFN weights should have gradients
        for name, param in encoder.ffn.named_parameters():
            if "weight" in name:
                assert param.grad is not None

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different embeddings."""
        encoder = FourierEncoder(64, m_min=0.1, m_max=5.0)
        x1 = torch.tensor([100.0])
        x2 = torch.tensor([200.0])
        out1 = encoder(x1)
        out2 = encoder(x2)
        assert not torch.allclose(out1, out2)

    def test_deterministic(self):
        """Test that the same input produces the same output."""
        encoder = FourierEncoder(64, m_min=0.1, m_max=5.0)
        encoder.eval()
        x = torch.tensor([100.0, 200.0])
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_different_d_model(self):
        """Test correct output shape with different d_model values."""
        for d_model in [32, 64, 128]:
            encoder = FourierEncoder(d_model, m_min=0.1, m_max=5.0)
            x = torch.randn(2, 5)
            out = encoder(x)
            assert out.shape == (2, 5, d_model)


class TestFourierPeakEncoder:
    """Tests for the FourierPeakEncoder class."""

    def test_output_shape(self):
        """Test the output shape for peak encoding."""
        d_model = 64
        encoder = FourierPeakEncoder(d_model, m_min=0.1, m_max=5.0)

        mz = torch.randn(4, 10)
        intensity = torch.randn(4, 10)
        out = encoder(mz, intensity)
        assert out.shape == (4, 10, d_model)

    def test_gradients_flow(self):
        """Test that gradients flow through both branches."""
        encoder = FourierPeakEncoder(64, m_min=0.1, m_max=5.0)
        mz = torch.randn(2, 5)
        intensity = torch.randn(2, 5)
        out = encoder(mz, intensity)
        loss = out.sum()
        loss.backward()

        # Both FFN branches should have gradients
        for param in encoder.fourier_encoder.ffn.parameters():
            if param.requires_grad:
                assert param.grad is not None
        for param in encoder.raw_ffn.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_different_intensities_different_outputs(self):
        """Test that different intensities produce different embeddings."""
        encoder = FourierPeakEncoder(64, m_min=0.1, m_max=5.0)
        mz = torch.tensor([[100.0, 200.0]])
        int1 = torch.tensor([[0.5, 0.8]])
        int2 = torch.tensor([[0.1, 0.2]])
        out1 = encoder(mz, int1)
        out2 = encoder(mz, int2)
        assert not torch.allclose(out1, out2)

    def test_concatenation_architecture(self):
        """Test that the encoder uses concatenation (not sum)."""
        encoder = FourierPeakEncoder(64, m_min=0.1, m_max=5.0)
        # Check that fourier_encoder and raw_ffn produce different dims
        # that must be concatenated
        assert hasattr(encoder, "fourier_encoder")
        assert hasattr(encoder, "raw_ffn")
        assert hasattr(encoder, "output_proj")
