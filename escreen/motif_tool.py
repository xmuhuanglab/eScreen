"""Motif (PWM) utilities for the eScreen sequence-prior front end.

This module builds the frozen, interpretable first stage of the eScreen
backbone described in Supplementary Methods section 2.3.6: 637 curated TF
archetype PWMs (Vierstra et al., MEME format, shipped as
``data/consensus_pwms.meme``) are parsed, padded to a common length, extended
with their reverse complements (637 -> 1,274 filters, one per motif and
strand), mean-centred, and transposed to the ``nn.Conv1d.weight`` layout
``(out_channels, in_channels=4, kernel_size=L)`` so they can be assigned
directly to a convolution whose weights are then frozen.

What it provides
----------------
``parse_meme_file``
    Minimal MEME reader -> ``(n_motifs, L, 4)`` array + names, tail-padded to
    29 bp.
``normalize_motif``
    Per-position mean centring of a PWM block.
``move_zeros_to_end``
    Left-compacts the non-zero part of each row (used for tail-padded PWMs).
``load_pwm``
    Reader + reverse complements + mean centring, returned as a float tensor
    ready for ``nn.Conv1d``.
``load_pwm_from_meme_c`` / ``load_pwm_from_meme_c1``
    Alternative readers that keep the forward and reverse-complement blocks
    separate, apply a log-odds-like ``p - 0.25`` transform plus scaling, and pad
    either at the tail (``_c``) or symmetrically (``_c1``).  These are the
    variants the model code uses; ``escreen/serve.py`` calls
    ``load_pwm_from_meme_c(..., max_length=35)``, i.e. the 35 bp padding of
    Supplementary Methods section 2.3.6, whereas the default here is 29.
``CenteredMaxPool1D``
    Odd-kernel max-pool with ``replicate`` padding that keeps both the sequence
    length and the positional alignment of a motif hit.

Role in the pipeline
--------------------
The ``(1274, 4, 35)`` tensor produced here becomes the weight of the motif
embedding layer; its 1,274 output channels each correspond deterministically to
one motif-strand combination, and their activations are projected into the
256-dimensional latent space that feeds the StripedHyena2 encoder.  Because the
weights are frozen, the channel-to-motif correspondence survives training, which
is what makes the attribution analysis of Supplementary Methods section 3
tractable.

Notes
-----
* This is a partly duplicated sibling of :mod:`escreen.motif`; both implement
  the same transforms with slightly different defaults and wrappers.  Both are
  re-exported from ``escreen/__init__.py``.
* All readers assume a nine-line MEME header and a column order of A, C, G, T.
* Tensors returned by ``load_pwm`` are CPU float32; nothing here touches a GPU
  or requires a CUDA build of torch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_meme_file(file_path):
    """Parse a MEME-format motif file into a stacked probability matrix.

    Implements the "load the 637 PWMs and pad each of them" step of
    Supplementary Methods section 2.3.6.

    Parameters
    ----------
    file_path : str or pathlib.Path
        MEME text file.  The reader skips a fixed nine-line header, so the first
        ``MOTIF`` record must start on line 10 of the file.

    Returns
    -------
    motifs_matrix : numpy.ndarray
        Shape ``(n_motifs, 29, 4)``, dtype float64.  Every motif occupies the
        same height because shorter motifs are zero-padded at the *tail* (the
        information stays at the 5' end of the kernel).
    motif_names : list of str
        Second whitespace token of each ``MOTIF`` line, e.g.
        ``'AC0001:GATA/PROP:GATA'``.

    Notes
    -----
    * Each probability row is divided by the motif width before stacking (see
      the inline comment), so the stored values are ``p / w`` rather than raw
      probabilities.  The factor is width dependent, so it does not cancel
      across motifs.
    * ``np.stack`` needs a rectangular array: a motif wider than 29 bp is not
      truncated and raises ``ValueError``.  The shipped
      ``data/consensus_pwms.meme`` has 637 motifs of width 4-28, so this never
      triggers there.
    * The ``alphabet`` read from the header is unused; A, C, G, T column order is
      hard-coded.
    * Motif records whose ``letter-probability matrix`` block ends the file are
      still flushed, because of the trailing append after the loop.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    current_motif = None
    # Header convention: line index 2 (0-based) of a MEME file is
    # 'ALPHABET= ACGT'.  The value is parsed but never used - the column order is
    # assumed to be A, C, G, T throughout this module.
    alphabet = lines[2].split()[1]
    # Skip the fixed nine-line header (version, blank, ALPHABET, blank,
    # strands, blank, two background lines, blank) so that the loop starts at the
    # first MOTIF record.  Any extra header line shifts every offset below and
    # silently corrupts the parse instead of raising.
    lines = lines[9:]
    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
            current_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_motif["width"] = len(letter_prob_matrix)
            # This divides the PWM by its WIDTH, not by a per-position
            # normaliser, so the stored values are p / w and the scale depends
            # on the motif length.
            current_motif["letter_prob_matrix"] /= current_motif["width"]
            # Pad short motifs at the tail so that all PWMs share a height of
            # 29 bp and can be stacked.  Motifs of 29 bp or more are left
            # untouched (and would break the np.stack below if wider than 29).
            if current_motif["width"] < 29:
                current_motif["letter_prob_matrix"] = np.concatenate(
                    (
                        current_motif["letter_prob_matrix"],
                        np.zeros((29 - current_motif["width"], 4)),
                    ),
                    axis=0,
                )
    if current_motif is not None:
        motifs.append(current_motif)

    motifs_matrix = np.stack([motif["letter_prob_matrix"]
                             for motif in motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]
    return motifs_matrix, motif_names

def normalize_motif(x):
    """Subtract each motif's per-position mean score (mean centring).

    Parameters
    ----------
    x : torch.Tensor
        PWM block of shape ``(n_motifs, L, 4)`` (motif index, position,
        nucleotide channel in A, C, G, T order), float.

    Returns
    -------
    torch.Tensor
        ``x - x.mean(dim=2, keepdim=True)``, same shape and dtype.  Every
        position row sums to zero, so a uniform background contributes a score
        of 0.

    Notes
    -----
    The per-position standard deviation is computed but never used: the return
    value is mean-centred, not standardised.  This is the log-odds behaviour of
    a 0.25 background without taking a logarithm.
    """
    mean = x.mean(dim=2, keepdim=True)
    std = x.std(dim=2, keepdim=True)
    return x-mean

def move_zeros_to_end(arr):
    """Left-compact the non-zero entries of each row, zero-filling the tail.

    Parameters
    ----------
    arr : numpy.ndarray
        2-D array of shape ``(n_rows, n_cols)``.

    Returns
    -------
    numpy.ndarray
        Same shape and dtype.  For every row, the non-zero entries are copied in
        their original order to the front and the remaining cells are zeroed.

    Notes
    -----
    Used to re-align tail-padded PWMs so that all motifs start at the same
    position.  Exact zeros are treated as padding, so a genuine 0.0 probability
    inside a motif would also be moved - acceptable here because the readers
    add no zero probabilities except as padding.
    """
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        non_zero_indices = np.where(arr[i] != 0)[0]
        result[i, :len(non_zero_indices)] = arr[i, non_zero_indices]
    
    return result

def load_pwm(pwm,include_reverse_complement=True):

    """Load PWMs as a Conv1d kernel, with reverse complements appended.

    Parameters
    ----------
    pwm : str or pathlib.Path
        MEME file path; see :func:`parse_meme_file`.
    include_reverse_complement : bool, default True
        Accepted for API compatibility but **not consulted**: the reverse
        complement block is always appended.

    Returns
    -------
    kernel : torch.Tensor
        Shape ``(2 * n_motifs, 4, L)``, dtype float32, laid out for
        ``nn.Conv1d(in_channels=4, out_channels=2 * n_motifs, kernel_size=L)``.
        With the shipped 637-motif file this is ``(1274, 4, 29)``, matching the
        1,274 motif filters of Supplementary Methods section 2.3.6.
    motif_names : list of str
        Forward-strand names only, length ``n_motifs``; the appended
        reverse-complement channels have no name.

    Notes
    -----
    * The reverse complement is ``motifs[:, ::-1, ::-1]``: reverse the position
      axis and complement the nucleotide axis, which maps the column order
      A, C, G, T to T, G, C, A.
    * Mean centring is applied *after* concatenation, so forward and reverse
      strands get the identical transform.
    * The padding height is fixed at 29 by :func:`parse_meme_file`; call
      :func:`load_pwm_from_meme_c` directly for the 35 bp padding used in the
      paper's configuration.
    """
    motifs, motif_names = parse_meme_file(pwm)
    # Reverse complement: reverse the position axis and complement the
    # nucleotide axis (A,C,G,T -> T,G,C,A).  This is the step that turns the
    # 637 forward PWMs into the 1,274 filters of Supplementary Methods 2.3.6.
    motifs_rev = motifs[:, ::-1, ::-1].copy()

    motifs = torch.tensor(motifs)
    motifs_rev = torch.tensor(motifs_rev)
    # Channel order matters downstream: filters 0..n-1 are forward-strand and
    # filters n..2n-1 are reverse-strand; motif_names covers only the forward
    # half, so index i and i + n name the same motif on opposite strands.
    motifs = torch.cat([motifs, motifs_rev], dim=0)

    # Mean-centre (not z-score) so that a uniform background scores 0.
    motifs = normalize_motif(motifs)
    
    # (n_motifs, L, 4) -> (n_filters, 4, L): the nn.Conv1d.weight layout, with
    # in_channels = 4 (A, C, G, T) and kernel_size = L.
    return motifs.permute(0,2,1).float(), motif_names

import numpy as np

def load_pwm_from_meme_c(file_path, max_length=29):
    """Load PWMs as separate forward/reverse arrays with log-odds shifting and padding.

    This is the variant used by the model code - ``escreen/serve.py`` imports
    ``load_pwm_from_meme_c`` from :mod:`escreen.motif` with
    ``max_length=35`` - and it reproduces Supplementary Methods section 2.3.6:
    637 PWMs, each padded to ``max_length``, plus its reverse complement.

    Parameters
    ----------
    file_path : str or pathlib.Path
        MEME file path.  Same nine-line header assumption as
        :func:`parse_meme_file`.
    max_length : int, default 29
        Target motif height in bp.  The paper's configuration uses 35; the
        default here is 29, so pass 35 explicitly when reproducing the model.

    Returns
    -------
    motifs_matrix : numpy.ndarray
        Forward PWMs, shape ``(n_motifs, 4, max_length)``, float64.
    complementary_motifs_matrix : numpy.ndarray
        Reverse-complement PWMs, same shape, built as ``matrix[::-1, ::-1]``
        (positions reversed, nucleotides complemented).
    motif_names : list of str
        Last whitespace token of each ``MOTIF`` line.
    motif_length : list of int
        Original, unpadded width of each motif, in file order.

    Notes
    -----
    Per-motif transform, applied identically to the forward and reverse blocks:
    subtract the 0.25 uniform background, divide by the per-position maximum,
    divide by the motif width, then zero-pad at the tail.  The result is a scaled
    log-odds-like score whose best base per position maps to ``1 / width``; it is
    not a probability matrix and does not sum to one.
    * ``np.stack`` again requires every motif to fit in ``max_length``; wider
      motifs are not truncated and raise ``ValueError``.
    * ``current_motif['name']`` takes the last token of the ``MOTIF`` line while
      ``current_complementary_motif['name']`` takes the second token; for the
      shipped file both tokens are the same string.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    complementary_motifs = []
    motif_length = []
    current_motif = None
    current_complementary_motif = None
    alphabet = lines[2].split()[1]

    lines = lines[9:]
    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
                complementary_motifs.append(current_complementary_motif)
            current_motif = {"name": line.split()[-1], "letter_prob_matrix": []}
            current_complementary_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            complementary_letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            current_complementary_motif["width"] = current_motif["width"]
            motif_length.append(current_motif["width"])
            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
                complementary_letter_prob_matrix.append(row)
            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_complementary_motif["letter_prob_matrix"] = np.array(complementary_letter_prob_matrix)[::-1,::-1]
            
            current_motif["width"] = len(letter_prob_matrix)
            current_complementary_motif["width"] = current_motif["width"]
            
            # Log-odds-style shift: subtracting the uniform background (0.25)
            # makes a base at background frequency score 0, then the per-position
            # maximum is scaled to 1.
            current_motif["letter_prob_matrix"] -= 0.25
            current_motif["letter_prob_matrix"] /= current_motif["letter_prob_matrix"].max(axis=-1,keepdims=True)
            current_motif["letter_prob_matrix"] /= current_motif["width"]
            
            current_complementary_motif["letter_prob_matrix"] -= 0.25
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["letter_prob_matrix"].max(axis=-1,keepdims=True)
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["width"]

            # Tail padding: zeros are appended after the last motif position,
            # so the motif sits at the 5' end of the kernel.  max_length must be
            # at least the widest motif (28 for the shipped file, 35 in the
            # paper's configuration).
            if current_motif["width"] < max_length:
                pad_length = max_length - current_motif["width"]
                current_motif["letter_prob_matrix"] = np.concatenate(
                    ( current_motif["letter_prob_matrix"], np.zeros((pad_length, 4)), ),
                    axis=0,
                )
                current_complementary_motif["letter_prob_matrix"] = np.concatenate(
                    ( current_complementary_motif["letter_prob_matrix"], np.zeros((pad_length, 4)), ),
                    axis=0,
                )
    if current_motif is not None:
        motifs.append(current_motif)
        complementary_motifs.append(current_complementary_motif)

    motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    complementary_motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in complementary_motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]
    
    # (n_motifs, L, 4) -> (n_motifs, 4, L): Conv1d weight layout.
    motifs_matrix=motifs_matrix.transpose(0,2,1)
    complementary_motifs_matrix=complementary_motifs_matrix.transpose(0,2,1)
    
    return motifs_matrix, complementary_motifs_matrix, motif_names, motif_length


import numpy as np

def load_pwm_from_meme_c1(file_path, max_length=29):
    """Same as :func:`load_pwm_from_meme_c` but with *symmetric* padding.

    Identical transform (0.25 background shift, per-position max scaling,
    width scaling), except that motifs shorter than ``max_length`` are centred
    inside the kernel instead of being pushed to the 5' end: ``pad_top =
    pad_total // 2`` zeros are inserted before the first position and the rest
    after the last.  Centring is what keeps the kernel's receptive field
    symmetric around the motif when the convolution is used for motif scanning
    at single-base resolution.

    Parameters
    ----------
    file_path : str or pathlib.Path
        MEME file path; nine-line header assumed.
    max_length : int, default 29
        Target motif height in bp.

    Returns
    -------
    motifs_matrix : numpy.ndarray
        Forward PWMs, shape ``(n_motifs, 4, max_length)``, float64.
    complementary_motifs_matrix : numpy.ndarray
        Reverse-complement PWMs, same shape.
    motif_names : list of str
        Last whitespace token of each ``MOTIF`` line.
    motif_length : list of int
        Original width of each motif, in file order.

    Notes
    -----
    * Because both flanks are padded with zeros, the 5' padding contributes a
      constant 0 to the score, which is harmless for the mean-centred /
      log-odds interpretation but does mean the effective kernel spans
      ``max_length`` bases even for a 4 bp motif.
    * ``motif_length`` is recorded before padding, so it is the only reliable
      way to recover where the real motif starts inside the padded kernel.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    complementary_motifs = []
    motif_length = []
    current_motif = None
    current_complementary_motif = None
    alphabet = lines[2].split()[1]
    lines = lines[9:]

    for i, line in enumerate(lines):
        if line.startswith("MOTIF"):
            if current_motif is not None:
                motifs.append(current_motif)
                complementary_motifs.append(current_complementary_motif)
            current_motif = {"name": line.split()[-1], "letter_prob_matrix": []}
            current_complementary_motif = {"name": line.split()[1], "letter_prob_matrix": []}
        elif current_motif is not None and line.startswith("letter-probability matrix"):
            letter_prob_matrix = []
            complementary_letter_prob_matrix = []
            current_motif["width"] = int(line.split()[5])
            current_complementary_motif["width"] = current_motif["width"]
            motif_length.append(current_motif["width"])

            for j in range(current_motif["width"]):
                row = list(map(float, lines[i + j + 1].split()))
                letter_prob_matrix.append(row)
                complementary_letter_prob_matrix.append(row)

            current_motif["letter_prob_matrix"] = np.array(letter_prob_matrix)
            current_complementary_motif["letter_prob_matrix"] = np.array(complementary_letter_prob_matrix)[::-1, ::-1]

            current_motif["width"] = len(letter_prob_matrix)
            current_complementary_motif["width"] = current_motif["width"]

            current_motif["letter_prob_matrix"] -= 0.25
            current_motif["letter_prob_matrix"] /= current_motif["letter_prob_matrix"].max(axis=-1, keepdims=True)
            current_motif["letter_prob_matrix"] /= current_motif["width"]

            current_complementary_motif["letter_prob_matrix"] -= 0.25
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["letter_prob_matrix"].max(axis=-1, keepdims=True)
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["width"]

            if current_motif["width"] < max_length:
            # Symmetric (centred) padding, unlike load_pwm_from_meme_c which
            # pads only at the tail.  When pad_total is odd the extra zero goes
            # after the motif (pad_bottom = pad_total - pad_top).
                pad_total = max_length - current_motif["width"]
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top
                current_motif["letter_prob_matrix"] = np.pad(
                    current_motif["letter_prob_matrix"],
                    ((pad_top, pad_bottom), (0, 0)),
                    mode='constant'
                )
                current_complementary_motif["letter_prob_matrix"] = np.pad(
                    current_complementary_motif["letter_prob_matrix"],
                    ((pad_top, pad_bottom), (0, 0)),
                    mode='constant'
                )

    if current_motif is not None:
        motifs.append(current_motif)
        complementary_motifs.append(current_complementary_motif)

    motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    complementary_motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in complementary_motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]

    motifs_matrix = motifs_matrix.transpose(0, 2, 1)
    complementary_motifs_matrix = complementary_motifs_matrix.transpose(0, 2, 1)

    return motifs_matrix, complementary_motifs_matrix, motif_names, motif_length

class CenteredMaxPool1D(nn.Module):
    """Odd-kernel 1-D max-pool that preserves length and motif centres.

    Wraps ``F.max_pool1d`` with symmetric ``replicate`` padding of
    ``kernel_size // 2`` on each side, so an input of shape ``(N, C, L)`` maps to
    an output with the length it would have under a ``padding='same'``
    convolution.  Used after the motif convolution when only the strongest
    activation in a small window matters (max over neighbouring positions)
    rather than the exact position of a hit.

    Parameters
    ----------
    kernel_size : int
        Pooling window in nucleotides; must be odd so that a unique centre
        position exists.
    stride : int, default 1
        Pooling stride; with the default the output length equals the input
        length.

    Notes
    -----
    * The odd-kernel requirement is asserted in ``__init__``, i.e. a
      construction-time check rather than a runtime shape check.
    * ``replicate`` padding keeps negative activations comparable at the
      borders; see ``forward``.
    """
    def __init__(self, kernel_size: int, stride: int = 1):
        """Store the pooling geometry and derive the symmetric padding width.

        Parameters
        ----------
        kernel_size : int
            Pooling window in nucleotides; must be odd.
        stride : int, default 1
            Pooling stride; with the default the output length equals the input
            length.

        Raises
        ------
        AssertionError
            If ``kernel_size`` is even.
        """
        super().__init__()
        # Odd kernels have a well-defined centre, which is what keeps the pooled
        # output aligned with the input; an even kernel would shift the signal by
        # half a base per layer.
        assert kernel_size % 2 == 1, "kernel_size must be an Odd number"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, x):
        """Apply centred max-pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(N, C, L)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, C, L_out)`` with
            ``L_out = floor((L + 2 * (kernel_size // 2) - kernel_size) / stride) + 1``,
            which equals ``L`` for the default ``stride=1``.

        Notes
        -----
        Padding is applied to both ends before pooling, so the first and last
        output positions are centred on the first and last input positions
        instead of being dropped.
        """
        # 'replicate' (edge) padding rather than zeros: with zero padding, a
        # window whose real activations are all negative would be dominated by
        # the padded zeros at the borders.
        padded_x = F.pad(x, (self.padding, self.padding), mode='replicate')
        pooled = F.max_pool1d(padded_x, kernel_size=self.kernel_size, stride=self.stride)
        return pooled