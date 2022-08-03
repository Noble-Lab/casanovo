import pytest
import torch
from depthcharge.components.transformers import (
    SpectrumEncoder,
    PeptideEncoder,
    PeptideDecoder,
)
from depthcharge.masses import PeptideMass


def test_spectrum_encoder():
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )
    model = SpectrumEncoder(
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        dim_intensity=None,
    )
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 128)
    assert mask.sum() == 1

    model = SpectrumEncoder(
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        dim_intensity=4,
    )
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 128)
    assert mask.sum() == 1

    model = SpectrumEncoder(8, 1, 12, dim_intensity=None)
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1

    model = SpectrumEncoder(8, 1, 12, dim_intensity=4)
    emb, mask = model(spectra)
    assert emb.shape == (2, 4, 8)
    assert mask.sum() == 1


def test_peptide_decoder():
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )
    peptides = ["LESLIEK", "PEPTIDER"]
    precursors = torch.tensor([[100.0, 2, 51.007276], [200.0, 3, 67.6739427]])
    dec_model = PeptideDecoder(
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        residues="canonical",
        max_charge=5,
    )
    enc_model = SpectrumEncoder(
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        dim_intensity=None,
    )
    memory, mem_mask = enc_model(spectra)
    scores, tokens = dec_model(peptides, precursors, memory, mem_mask)
    assert scores.shape == (2, 10, dec_model.vocab_size + 1)
    assert tokens.shape == (2, 9)
