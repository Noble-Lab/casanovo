"""Fourier positional encodings for mass spectrometry data."""

import math

import torch


def _build_fourier_frequencies(
    m_min: float = 1e-4,
    m_max: float = 1000.0,
) -> torch.Tensor:
    """Build the predefined Fourier frequency vector.

    Constructs frequencies as described in the DreaMS paper (Eq. 4):

    - Low frequencies:  1/m_max, 1/(m_max-1), ..., 1
       (captures integer part of mass)
    - High frequencies: 1/(k*m_min), 1/((k-1)*m_min), ..., 1/m_min
       (captures decimal part of mass)

    Parameters:
    m_min : float, optional
        Minimum decimal mass of interest (absolute instrument accuracy).
        Default is 1e-4 as used for GeMS datasets.
    m_max : float, optional
        Maximum integer mass of interest.
        Default is 1000 as used for GeMS datasets.

    Returns:
    torch.Tensor
        The 1D frequency vector b of shape (B,).
    """
    # Low frequencies: 1/m_max, 1/(m_max-1), ..., 1/1
    # These capture integer mass differences.
    low_denom = torch.arange(m_max, 0, -1, dtype=torch.float32)
    low_freqs = 1.0 / low_denom

    # High frequencies: 1/(k*m_min), ..., 1/m_min
    # k is such that k*m_min is closest to 1.
    k = round(1.0 / m_min)
    high_denom = torch.arange(k, 0, -1, dtype=torch.float32) * m_min
    high_freqs = 1.0 / high_denom

    return torch.cat([low_freqs, high_freqs])


class FourierEncoder(torch.nn.Module):
    """Encode floating-point values using predefined Fourier features.

    Implements the Fourier feature encoding from the DreaMS paper:

        Φ(m)_i     = sin(2π · b_i · m)
        Φ(m)_{i+1} = cos(2π · b_i · m)

    where b is a vector of predefined frequencies split into low
    (integer mass) and high (decimal mass) components, followed by
    a feed-forward network to project from 2B dimensions to d_model.

    Parameters:
    d_model : int
        The output embedding dimensionality.
    m_min : float, optional
        Minimum decimal mass of interest (instrument accuracy).
    m_max : float, optional
        Maximum integer mass of interest.
    n_layers_ffn : int, optional
        Number of hidden layers in the projection FFN.
    """

    def __init__(
        self,
        d_model: int = 128,
        m_min: float = 1e-4,
        m_max: float = 1000.0,
        n_layers_ffn: int = 1,
    ) -> None:
        """Initialize a FourierEncoder."""
        super().__init__()

        # Build and register the predefined frequencies (not learned).
        freqs = _build_fourier_frequencies(m_min, m_max)
        self.register_buffer("frequencies", freqs)
        n_fourier = 2 * len(freqs)  # sin + cos for each frequency

        # Feed-forward network to project Fourier features to d_model.
        layers = []
        in_dim = n_fourier
        for _ in range(n_layers_ffn):
            layers.append(torch.nn.Linear(in_dim, d_model))
            layers.append(torch.nn.GELU())
            in_dim = d_model
        layers.append(torch.nn.Linear(in_dim, d_model))
        self.ffn = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode floating-point values with Fourier features.

        Parameters:
        x : torch.Tensor of shape (...)
            Floating-point values to encode (e.g. masses or m/z).

        Returns:
        torch.Tensor of shape (..., d_model)
            The Fourier embeddings projected to d_model dimensions.
        """
        # x: (...,) -> (..., 1) for broadcasting with frequencies
        x = x.unsqueeze(-1).float()
        # (..., B) angles
        angles = 2.0 * math.pi * self.frequencies * x
        # Concatenate sin and cos -> (..., 2B)
        fourier_features = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )
        # Project to d_model via FFN -> (..., d_model)
        return self.ffn(fourier_features)


class FourierPeakEncoder(torch.nn.Module):
    """Encode (m/z, intensity) peak pairs using Fourier features.

    Implements the PeakEncoder from the DreaMS paper (Eq. 5):

        PeakEncoder(m, i) = FFN_F(Φ(m)) || FFN_P(m, i)

    The m/z values are encoded via Fourier features followed by FFN_F,
    and the raw m/z + intensity are processed by a separate FFN_P. The
    outputs are concatenated and projected to d_model.

    Parameters:
    d_model : int
        The output embedding dimensionality.
    m_min : float, optional
        Minimum decimal mass of interest (instrument accuracy).
    m_max : float, optional
        Maximum integer mass of interest.
    """

    def __init__(
        self,
        d_model: int = 128,
        m_min: float = 1e-4,
        m_max: float = 1000.0,
    ) -> None:
        """Initialize a FourierPeakEncoder."""
        super().__init__()
        self.d_model = d_model

        # FFN_F: Fourier features of m/z -> d_m dimensional
        d_fourier_out = d_model // 2
        self.fourier_encoder = FourierEncoder(d_fourier_out, m_min, m_max)

        # FFN_P: raw (m/z, intensity) -> d_p dimensional
        d_raw_out = d_model - d_fourier_out
        self.raw_ffn = torch.nn.Sequential(
            torch.nn.Linear(2, d_raw_out),
            torch.nn.GELU(),
            torch.nn.Linear(d_raw_out, d_raw_out),
        )

        # Final projection after concatenation.
        self.output_proj = torch.nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Encode m/z and intensity pairs.

        Parameters:
        x : torch.Tensor of shape (n_spectra, n_peaks)
            The m/z values for each peak.
        y : torch.Tensor of shape (n_spectra, n_peaks)
            The intensity values for each peak.

        Returns:
        torch.Tensor of shape (n_spectra, n_peaks, d_model)
            The encoded peaks.
        """
        # FFN_F(Φ(m/z))  -> (n_spectra, n_peaks, d_fourier_out)
        fourier_emb = self.fourier_encoder(x)

        # FFN_P(m/z, intensity) -> (n_spectra, n_peaks, d_raw_out)
        raw_input = torch.stack([x.float(), y.float()], dim=-1)  # (..., 2)
        raw_emb = self.raw_ffn(raw_input)

        # Concatenate and project -> (n_spectra, n_peaks, d_model)
        combined = torch.cat([fourier_emb, raw_emb], dim=-1)
        return self.output_proj(combined)
