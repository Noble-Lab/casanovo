"""Transformer encoder and decoder for the de novo sequencing task."""

import warnings
from collections.abc import Callable, Sequence

import torch
from depthcharge.encoders import FloatEncoder, PeakEncoder, PositionalEncoder
from depthcharge.tokenizers import Tokenizer
from depthcharge.transformers import (
    AnalyteTransformerDecoder,
    SpectrumTransformerEncoder,
)


class PeptideDecoder(AnalyteTransformerDecoder):
    """
    A transformer decoder for peptide sequences.

    Parameters
    ----------
    n_tokens : int
        The number of tokens used to tokenize peptide sequences.
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass
        spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``nhead``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the
        Transformer layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    positional_encoder : PositionalEncoder or bool, optional
        The positional encodings to use for the amino acid sequence. If
        ``True``, the default positional encoder is used. ``False``
        disables positional encodings, typically only for ablation
        tests.
    padding_int : int or None, optional
        The index that represents padding in the input sequence.
        Required only if ``n_tokens`` was provided as an ``int``.
    max_charge : int, optional
        The maximum charge state for peptide sequences.
    """

    def __init__(
        self,
        n_tokens: int | Tokenizer,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        padding_int: int | None = None,
        max_charge: int = 4,
        inter_ctc_layers: Sequence[int] = (),
    ) -> None:
        """Initialize a PeptideDecoder."""

        super().__init__(
            n_tokens=n_tokens,
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            positional_encoder=positional_encoder,
            padding_int=padding_int,
        )

        self.charge_encoder = torch.nn.Embedding(max_charge, d_model)
        self.mass_encoder = FloatEncoder(d_model)

        # Override the output layer with one class beyond the token
        # embeddings (which include padding at index 0): the last index
        # serves as the dedicated CTC blank class.
        self.final = torch.nn.Linear(
            d_model, self.token_encoder.num_embeddings + 1
        )

        # The layers after which this decoder also scores its own hidden
        # states, for an auxiliary CTC loss each. Scoring reuses the output
        # layer, so no layer here grows the model, and a checkpoint trained
        # without them still loads.
        requested = tuple(sorted(set(inter_ctc_layers)))
        self.inter_ctc_layers = tuple(
            k for k in requested if 1 <= k < n_layers
        )
        dropped = [k for k in requested if k not in self.inter_ctc_layers]
        if dropped:
            warnings.warn(
                f"Ignoring inter_ctc_layers {dropped}: a decoder of "
                f"{n_layers} layers can score layers 1 to {n_layers - 1}. "
                f"Scoring {list(self.inter_ctc_layers)}.",
                stacklevel=2,
            )

    def forward_with_intermediates(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Decode, scoring the stack's own hidden states along the way.

        Intermediate CTC (Lee and Watanabe, ICASSP 2021): after each layer
        in ``inter_ctc_layers`` the hidden states are scored with the same
        output layer the model already has, and that prediction carries an
        auxiliary CTC loss during training. The score is not fed back into
        the stack, so the layers below are supervised without changing what
        the layers above receive, and what a spectrum decodes to is
        unaffected. Inference still runs this method and scores those
        layers, for predictions nothing reads.

        This has to open up the layer stack rather than call
        ``transformer_decoder`` in one shot, so it prepares the frames
        itself with ``_prepare_frames``.

        Returns
        -------
        scores : torch.Tensor of shape (batch, len_seq, n_tokens)
            The final-layer scores, identical in meaning to ``forward``.
        intermediates : list of torch.Tensor
            One score tensor per layer in ``inter_ctc_layers``, for the
            auxiliary CTC losses. Empty when the list is, in which case
            the scores match ``forward`` exactly.
        """
        encoded, tgt_mask = self._prepare_frames(tokens, *args, **kwargs)
        intermediates = []
        for depth, layer in enumerate(self.transformer_decoder.layers, 1):
            encoded = layer(
                encoded,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if depth in self.inter_ctc_layers:
                intermediates.append(self.final(encoded))

        if self.transformer_decoder.norm is not None:
            encoded = self.transformer_decoder.norm(encoded)
        return self.final(encoded), intermediates

    def global_token_hook(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Override global_token_hook to include precursor information.

        Parameters
        ----------
        *args :
        tokens : list of str, torch.Tensor, or None
            The partial molecular sequences for which to predict the
            next token. Optionally, these may be the token indices
            instead of a string.
        precursors : torch.Tensor
            Precursor information.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The global token representations.
        """
        masses = self.mass_encoder(precursors[:, None, 0]).squeeze(1)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges
        return precursors

    def _prepare_frames(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the decoder frames and build the all-False target mask.

        The mask lets every frame see every other frame. No key padding
        mask goes with it: the superclass infers padding as
        ``encoded.sum(axis=2) == 0``, and every frame here is fed token
        id 0, ``padding_idx``, whose embedding is zero. That marked every
        frame as padding, so no frame could attend to another, when none
        of them is padding.
        """
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)
        encoded = self.token_encoder(tokens)
        global_token = self.global_token_hook(tokens, *args, **kwargs)
        encoded = torch.cat([global_token[:, None, :], encoded], dim=1)
        encoded = self.positional_encoder(encoded)
        length = encoded.shape[1]
        tgt_mask = torch.zeros(
            (length, length), dtype=torch.bool, device=encoded.device
        )
        return encoded, tgt_mask

    def embed(
        self,
        tokens: torch.Tensor | None,
        *args: torch.Tensor,
        memory: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Embed the decoder frames with full, non-causal attention.

        Reimplements the superclass rather than delegating to it, so
        that no key padding mask reaches the layers unless a caller asks
        for one (see ``_prepare_frames``).

        ``tgt_key_padding_mask`` marks padding in ``tokens``, and the
        precursor token is prepended here so callers need not. Leave it
        None to decode de novo, where every frame is a real slot; pass
        ``tokens == padding_idx`` when the tokens are real sequences
        padded to a common length, as in a database search. Nothing can
        infer which case applies, since both pad with token 0.
        """
        encoded, full_mask = self._prepare_frames(tokens, *args, **kwargs)
        if tgt_key_padding_mask is not None:
            # Position 0 holds the precursor token, never padding.
            tgt_key_padding_mask = torch.cat(
                [
                    torch.zeros(
                        (tgt_key_padding_mask.shape[0], 1),
                        dtype=torch.bool,
                        device=tgt_key_padding_mask.device,
                    ),
                    tgt_key_padding_mask,
                ],
                dim=1,
            )
        return self.transformer_decoder(
            tgt=encoded,
            memory=memory,
            tgt_mask=full_mask if tgt_mask is None else tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            memory_mask=memory_mask,
        )


class SpectrumEncoder(SpectrumTransformerEncoder):
    """
    A Transformer encoder for input mass spectra.

    Parameters
    ----------
    d_model : int, optional
        The latent dimensionality to represent peaks in the mass
        spectrum.
    n_head : int, optional
        The number of attention heads in each layer. ``d_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int, optional
        The dimensionality of the fully connected layers in the
        Transformer layers of the model.
    n_layers : int, optional
        The number of Transformer layers.
    dropout : float, optional
        The dropout probability for all layers.
    peak_encoder : PeakEncoder or bool, optional
        The function to encode the (m/z, intensity) tuples of each mass
        spectrum. `True` uses the default sinusoidal encoding and `False`
        instead performs a 1 to `d_model` learned linear projection.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable | bool = True,
    ):
        """Initialize a SpectrumEncoder."""
        super().__init__(
            d_model, n_head, dim_feedforward, n_layers, dropout, peak_encoder
        )

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: dict,
    ) -> torch.Tensor:
        """
        Override global_token_hook to include latent_spectrum parameter.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, max_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, max_peaks)
            The zero-padded intensity dimension for a batch of mass
            spectra.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.

        """
        return self.latent_spectrum.squeeze(0).expand(mz_array.shape[0], -1)
