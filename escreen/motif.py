"""Motif PWM loading and pre-processing for the eScreen input stage.

Provides the position-weight matrices that initialise (and then freeze) the
first convolutional layer of eScreen, as described in Supplementary Methods
Section 2.3.6: 637 transcription-factor PWMs taken from Vierstra et al. are
loaded from a MEME-format database, each is padded to a fixed length, and the
reverse complement of every motif is added so that a single ``Conv1d`` over the
one-hot sequence yields one channel per (motif, strand) pair.

Four loaders coexist here.  They differ only in their normalisation scheme and
in how they return the forward/reverse-complement pair:

``parse_meme_file``
    Minimal MEME parser; returns raw probability matrices only.
``load_pwm``
    Parser + reverse complement + centring.  Returns a single stacked tensor.
``load_pwm_from_meme_c``
    Min-max normalised forward and reverse-complement matrices, returned
    separately together with the original motif lengths.
``load_pwm_from_meme_c1``
    Normalisation selectable via ``norm`` (``'minmax'`` or ``'log'``); the
    ``'log'`` mode additionally fits a logistic curve to the PWM score
    distribution and returns the fitted scale and bias per motif, so that the
    motif score can later be turned into a probability.

IMPORTANT discrepancy with the manuscript
-----------------------------------------
Section 2.3.6 states that each PWM is padded to a **(4, 35)** matrix.  The
loaders below default to ``max_length=29``, and the padding is appended at the
end (``parse_meme_file``, ``load_pwm_from_meme_c``) or centred
(``load_pwm_from_meme_c1``) rather than at a fixed offset.  The number of
convolution channels actually used therefore depends on the MEME file and on the
``max_length`` passed by the caller, not on a constant in this module.  This is
documented rather than silently "fixed", because changing the padding changes
the receptive field of the motif filters and hence the trained weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_meme_file(file_path):
    """Parse a MEME-format motif database into probability matrices.

    Parameters
    ----------
    file_path : str
        Path to a MEME file.  The parser assumes the header block occupies the
        first nine lines (``MEME version``, ``ALPHABET``, ``strands``, the
        background line and the ``MOTIF`` section preamble), and that the
        nucleotide alphabet is described on line 3.

    Returns
    -------
    motifs_matrix : numpy.ndarray
        Array of shape ``[n_motifs, 29, 4]`` (or ``[n_motifs, max(width), 4]``
        if any motif is longer than 29), row-normalised and scaled.
    motif_names : list of str
        Motif identifiers, in file order.

    Notes
    -----
    Three details of this parser are easy to misread:

    * Only motifs shorter than 29 bp are padded, with **zero rows appended at
      the end**.  Longer motifs are left untouched, so the returned array can in
      principle be ragged-height-free only if all motifs fit.
    * ``letter_prob_matrix /= width`` divides by the motif width, i.e. it scales
      the whole matrix by a constant.  This is a deliberate down-scaling of the
      per-position probabilities, not a normalisation along any axis.
    * ``alphabet`` is parsed but never used; the code assumes the fixed
      ``A C G T`` column order throughout.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []
    current_motif = None
    alphabet = lines[2].split()[1]
    # skip the first 10 lines
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
            current_motif["letter_prob_matrix"] /= current_motif["width"]
            if current_motif["width"] < 29:
                current_motif["letter_prob_matrix"] = np.concatenate(
                    (
                        current_motif["letter_prob_matrix"],
                        np.zeros((29 - current_motif["width"], 4)),
                    ),
                    axis=0,
                )
    if current_motif is not None:
        # pad to 29, 4
        motifs.append(current_motif)

    motifs_matrix = np.stack([motif["letter_prob_matrix"]
                             for motif in motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]
    return motifs_matrix, motif_names

def normalize_motif(x):
    """Centre each motif along the nucleotide axis.

    Subtracts the per-position mean over the four nucleotides so that a motif
    row becomes a contrast vector (which base is preferred relative to the
    average base) instead of a probability vector.  This is what makes the
    frozen filters usable as a signed scoring function: a uniform row becomes
    all zeros and contributes nothing.

    Parameters
    ----------
    x : torch.Tensor
        Motif tensor, ``[n_motifs, length, 4]``.

    Returns
    -------
    torch.Tensor
        Centred tensor of the same shape.

    Notes
    -----
    ``std`` is computed on the line below but never used -- the function centres
    only and does **not** divide by the standard deviation.  The variable is
    dead; the behaviour is mean removal, not standardisation.
    """
    mean = x.mean(dim=2, keepdim=True)  # shape: (batch_size, 1, channels)
    std = x.std(dim=2, keepdim=True)    # shape: (batch_size, 1, channels)
    return x-mean

def move_zeros_to_end(arr):
    """Compact each row so that its non-zero entries are left-aligned.

    Padded motif matrices store the real positions first and the zero padding
    afterwards only when the padding was appended at the end.  This helper moves
    every non-zero value to the front of its row, which is needed when a matrix
    was padded in a way that interleaved zeros (e.g. centred padding).

    Parameters
    ----------
    arr : numpy.ndarray
        Two-dimensional array, ``[n_motifs, length]`` or ``[n_motifs, width]``.
        Each row is processed independently.

    Returns
    -------
    numpy.ndarray
        New array of the same shape with non-zero entries packed to the left and
        the remaining positions zero-filled.  The input is not modified.

    Notes
    -----
    A value of exactly ``0`` counts as padding.  For a centred motif in which a
    genuine nucleotide contrast happens to be exactly zero, that position would
    be treated as padding and shifted -- harmless for the frozen-filter use case,
    but not a lossless operation in general.
    """
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        non_zero_indices = np.where(arr[i] != 0)[0]
        result[i, :len(non_zero_indices)] = arr[i, non_zero_indices]
    
    return result

def load_pwm(pwm,include_reverse_complement=True):
    """Load a MEME file as a strand-concatenated, centred filter tensor.

    Parameters
    ----------
    pwm : str
        Path to a MEME-format motif database.
    include_reverse_complement : bool, default True
        Accepted for API compatibility but **never consulted**: the reverse
        complement is always appended, so the returned tensor always has
        ``2 * n_motifs`` channels.

    Returns
    -------
    motifs : torch.Tensor
        Filter tensor of shape ``[2 * n_motifs, 4, length]``, centred along the
        nucleotide axis by :func:`normalize_motif`.  The order is all forward
        motifs followed by all reverse complements, which is exactly the channel
        ordering expected by ``eSCREEN_backbone`` in ``'pwm'`` mode: channel
        ``i < n`` is motif ``i`` on the forward strand, channel ``i + n`` is the
        same motif on the reverse strand (Section 2.3.6).
    motif_names : list of str
        Motif identifiers, one per input motif (not duplicated for the reverse
        complements).

    Notes
    -----
    The reverse complement is built as ``motifs[:, ::-1, ::-1]``: reversing the
    position axis gives the opposite strand reading direction, and reversing the
    four-nucleotide axis performs the complement because the columns are ordered
    ``A C G T`` (index 0<->3, 1<->2).  This depends entirely on that column
    order; a different alphabet ordering would silently produce a wrong
    complement rather than raising.
    """

    
    # load pwm
    motifs, motif_names = parse_meme_file(pwm)
    motifs_rev = motifs[:, ::-1, ::-1].copy()
    
    # construct reverse complement
    motifs = torch.tensor(motifs)
    motifs_rev = torch.tensor(motifs_rev)
    motifs = torch.cat([motifs, motifs_rev], dim=0)
    
    # normalize motif
    motifs = normalize_motif(motifs)
    
    return motifs.permute(0,2,1).float(), motif_names

import numpy as np
from scipy.optimize import curve_fit

def init_dist(dmin, dmax, dp, weights, probs):
    """Build the score distribution of a single PWM column on a discrete grid.

    For one motif position the possible scores are the four nucleotide weights.
    Mapping each weight to a grid index and accumulating its probability gives a
    sparse discrete distribution over that position's score.

    Parameters
    ----------
    dmin, dmax : float
        Bounds of this column's score range, used to size the grid.
    dp : float
        Grid step (``gran`` in :func:`scoreDist`).
    weights : sequence of float
        The four per-nucleotide scores at this position.
    probs : sequence of float
        Probability of each nucleotide at this position, same length as
        ``weights``.

    Returns
    -------
    numpy.ndarray
        Density array of length ``round((dmax - dmin) / dp) + 1``.  Empty for a
        grid step of zero.

    Notes
    -----
    ``weights - dmin`` shifts the column minimum to index 0, so the returned
    array is indexed relative to ``dmin`` rather than absolutely.  When several
    nucleotides quantise to the same grid cell their probabilities are summed,
    which is what makes the convolution in :func:`scoreDist` well defined despite
    the discretisation.
    """
    out = np.zeros(int(np.round((dmax-dmin)/dp)+1))
    ii = np.array(np.round((weights-dmin)/dp), dtype=int)
    for i in range(len(probs)):
        out[ii[i]] = out[ii[i]] + probs[i]
    return out

def scoreDist(pwm, nucleotide_prob=None, gran=None, size=1000):
    """Exact score distribution of a PWM under a background nucleotide model.

    A PWM score is a sum of independent per-position contributions, so its
    distribution is the convolution of the per-position distributions.  This
    function builds those with :func:`init_dist` and convolves them, giving the
    null distribution of motif scores against a background.

    Parameters
    ----------
    pwm : numpy.ndarray
        Motif matrix, ``[width, 4]``, in log-odds or score space (not raw
        probabilities).
    nucleotide_prob : sequence of numpy.ndarray, optional
        Background probability of each nucleotide at each position, one 4-vector
        per row.  Defaults to uniform ``0.25`` at every position.
    gran : float, optional
        Grid step.  If omitted, derived from ``size`` as
        ``(max(pwm) - min(pwm)) / (size - 1)``.
    size : int, default 1000
        Number of grid points used to derive ``gran`` when ``gran`` is None.

    Returns
    -------
    support : numpy.ndarray
        Score values over which the distribution is non-zero.  Note the offset:
        the true minimum is the sum of the per-column minima, added back here.
    distribution : numpy.ndarray
        Corresponding probability masses, normalised to sum to 1.

    Notes
    -----
    The PWM is first quantised to the grid (``round(pwm / gran) * gran``) so that
    the convolution stays on a common index lattice.  Quantisation is lossy: two
    distinct scores closer than ``gran`` collapse, which is a deliberate
    accuracy/memory trade-off -- the caller controls it through ``size``.

    Only positive-mass entries are returned (``distribution > 0``), so ``support``
    is generally shorter than the internal grid.  Consumers that build a CDF, such
    as the logistic fit in :func:`load_pwm_from_meme_c1`, rely on this being a
    compact, sorted list.
    """
    if nucleotide_prob is None:
        nucleotide_prob = [np.ones(4)/4]*pwm.shape[0]
    if gran is None:
        if size is None:
            raise ValueError("provide either gran or size. Both missing.")
        gran = (np.max(pwm) - np.min(pwm))/(size - 1)
    pwm = np.round(pwm/gran)*gran
    pwm_max, pwm_min = pwm.max(axis=1), pwm.min(axis=1)
    distribution = init_dist(pwm_min[0], pwm_max[0], gran, pwm[0], nucleotide_prob[0])   
    for i in range(1, pwm.shape[0]):
        kernel = init_dist(pwm_min[i], pwm_max[i], gran, pwm[i], nucleotide_prob[i])
        distribution = np.convolve(distribution, kernel)
    support_min = pwm_min.sum()
    ii = np.where(distribution > 0)[0]
    support = support_min + (ii) * gran
    return support, distribution[ii]

def load_pwm_from_meme_c(file_path, max_length=29):
    """Load a MEME file as separate forward and reverse-complement matrices.

    Unlike :func:`load_pwm`, the two strands are returned as individual arrays
    rather than concatenated, and the original motif widths are preserved so the
    caller can mask the zero padding afterwards.

    Parameters
    ----------
    file_path : str
        MEME-format motif database.
    max_length : int, default 29
        Padding length.  Note that Section 2.3.6 documents 35 bp; see the module
        docstring for the discrepancy.

    Returns
    -------
    motifs_matrix : numpy.ndarray
        Forward matrices, ``[n_motifs, 4, max_length]``.
    complementary_motifs_matrix : numpy.ndarray
        Reverse-complement matrices, same shape and ordering.
    motif_names : list of str
        Motif identifiers.
    motif_length : list of int
        Original (pre-padding) width of each motif.

    Notes
    -----
    The normalisation applied per motif is::

        p -= 0.25                     # subtract the uniform background
        p /= p.max(axis=-1)           # scale each column by its own maximum
        p /= width                    # divide the whole matrix by the motif width

    The middle step makes each position's strongest nucleotide score 1, while the
    last step makes the accumulated motif score comparable across motifs of
    different widths.  Padding rows are appended **at the end** as zeros, which
    is why ``motif_length`` is returned -- without it a consumer cannot tell a
    genuine zero from padding.

    ``alphabet`` is parsed from the header but unused, and the forward and
    reverse-complement names are taken from different fields
    (``split()[-1]`` vs ``split()[1]``); only the forward names are returned, so
    this asymmetry has no visible effect unless the callers are reworked.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    motifs = []  # store the forward matrices
    complementary_motifs = []  # store the reverse-complement matrices
    motif_length = [] # store the motif lengths
    current_motif = None
    current_complementary_motif = None
    alphabet = lines[2].split()[1]
    # skip the first 10 lines
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
            
            current_motif["letter_prob_matrix"] -= 0.25
            current_motif["letter_prob_matrix"] /= current_motif["letter_prob_matrix"].max(axis=-1,keepdims=True)
            current_motif["letter_prob_matrix"] /= current_motif["width"]
            
            current_complementary_motif["letter_prob_matrix"] -= 0.25
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["letter_prob_matrix"].max(axis=-1,keepdims=True)
            current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["width"]
            
            # pad to max_length
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
        # pad to max_length
        motifs.append(current_motif)
        complementary_motifs.append(current_complementary_motif)

    # stack the forward and reverse-complement matrices into numpy arrays
    motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    complementary_motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in complementary_motifs], axis=0)
    motif_names = [motif["name"] for motif in motifs]
    
    motifs_matrix=motifs_matrix.transpose(0,2,1)
    complementary_motifs_matrix=complementary_motifs_matrix.transpose(0,2,1)
    
    return motifs_matrix, complementary_motifs_matrix, motif_names, motif_length

def load_pwm_from_meme_c1(file_path, max_length=29, norm='minmax'):
    """Load a MEME file with a selectable normalisation and a fitted bias.

    The most general of the four loaders.  Besides the matrices it returns a
    per-motif bias term, so that a motif's raw convolution output can be mapped
    to a calibrated probability via ``sigmoid(score + bias)`` -- this is what the
    ``'log'`` mode exists for.

    Parameters
    ----------
    file_path : str
        MEME-format motif database.
    max_length : int, default 29
        Padding length; see the module docstring for the 29-vs-35 discrepancy
        against Section 2.3.6.
    norm : {'minmax', 'log', None}, default 'minmax'
        Normalisation strategy.

        ``'minmax'``
            Subtract 0.25, divide each column by its maximum, divide by width;
            bias is fixed at 0.  Equivalent to
            :func:`load_pwm_from_meme_c`.
        ``'log'``
            Convert to log space, build the exact score distribution with
            :func:`scoreDist`, and fit ``1 / (1 + exp(-(a*x + b)))`` to its CDF
            with :func:`scipy.optimize.curve_fit`.  The fitted ``a`` rescales the
            matrix and ``b`` becomes the bias, which makes the sigmoid of the
            motif score approximate the probability that the motif is present.
        ``None`` or ``'none'``
            No normalisation; matrices are returned raw.  ``bias`` is then never
            assigned, so the per-motif bias lookup below raises ``KeyError``.
        Any other value prints a warning and leaves the matrix raw (with the same
        missing-bias consequence).

    Returns
    -------
    forward : tuple of (numpy.ndarray, numpy.ndarray)
        ``(motifs_matrix, motifs_bias)``, matrix shape
        ``[n_motifs, 4, max_length]``.
    reverse : tuple of (numpy.ndarray, numpy.ndarray)
        The same for the reverse-complement strand.
    motif_names : list of str
        Motif identifiers.
    motif_length : list of int
        Original widths, before padding.

    Notes
    -----
    Padding here is **centred** (``pad_top = pad_total // 2``), unlike the
    end-padding used by the other loaders.  Centred padding keeps a motif's
    middle aligned with the filter's middle, which is the right choice when the
    filter is later used with ``padding='same'`` convolution.

    The ``'log'`` branch multiplies the probabilities by 4 before taking the log,
    i.e. ``log(4p)`` rather than the textbook log-odds ``log(p / 0.25)``.  These
    agree only when all four nucleotides share the same background, so the result
    should be read as a scaled log-probability rather than a strict log-odds
    score.

    ``bias`` is fetched for every motif in the final stacking step; the
    ``norm=None`` and unknown-norm paths never set it, so those paths fail at the
    end rather than at the point of the bad argument.
    """

    
    def logit_func(x, a, b):
        """Logistic curve fitted to the PWM score CDF.

        Used by the ``'log'`` normalisation branch to turn the empirical score
        distribution into a calibrated presence probability.  ``a`` rescales the
        motif score and ``b`` shifts it; both are fitted, then ``a`` is folded
        back into the matrix and ``b`` is returned as the per-motif bias.

        Parameters
        ----------
        x : array-like
            Score support produced by :func:`scoreDist`.
        a : float
            Fitted scale.
        b : float
            Fitted offset.

        Returns
        -------
        array-like
            ``sigmoid(a * x + b)``.
        """
        return 1 / (1 + np.exp(-(a * x + b)))
    
    with open(file_path, "r") as f:lines = f.readlines()

    motifs = []  # store the forward matrices
    complementary_motifs = []  # store the reverse-complement matrices
    motif_length = []  # store the motif lengths
    current_motif = None
    current_complementary_motif = None
    alphabet = lines[2].split()[1]
    lines = lines[9:]  # skip the first 9 lines

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

            # normalization
            if norm is None or norm == 'none':
                pass
            elif norm == 'minmax':
                current_motif["letter_prob_matrix"] -= 0.25
                current_motif["letter_prob_matrix"] /= current_motif["letter_prob_matrix"].max(axis=-1, keepdims=True)
                current_motif["letter_prob_matrix"] /= current_motif["width"]
                current_motif["bias"]               = 0

                current_complementary_motif["letter_prob_matrix"] -= 0.25
                current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["letter_prob_matrix"].max(axis=-1, keepdims=True)
                current_complementary_motif["letter_prob_matrix"] /= current_complementary_motif["width"]
                current_complementary_motif["bias"]               = 0
            elif norm == 'log':
                current_motif["letter_prob_matrix"] *= 4
                current_motif["letter_prob_matrix"] = np.where(current_motif["letter_prob_matrix"]>0,np.log(current_motif["letter_prob_matrix"]),0)
                nucleotide_prob = np.exp(current_motif["letter_prob_matrix"]) / np.sum(np.exp(current_motif["letter_prob_matrix"]), axis=1, keepdims=True)
                #nucleotide_prob = np.ones((current_motif["width"], 4)) * 0.25
                s,d = scoreDist(current_motif["letter_prob_matrix"], nucleotide_prob=nucleotide_prob, gran=None, size=1000)
                cdf = np.cumsum(d)
                params, _ = curve_fit(logit_func, s, cdf, maxfev=1000) # obtain the correction coefficients and bias by curve fitting
                w, b = params
                current_motif["letter_prob_matrix"] *= w
                current_motif["bias"]               = b
               
                current_complementary_motif["letter_prob_matrix"] *= 4
                current_complementary_motif["letter_prob_matrix"] = \
                    np.where(current_complementary_motif["letter_prob_matrix"]>0,np.log(current_complementary_motif["letter_prob_matrix"]),0)
                nucleotide_prob = \
                    np.exp(current_complementary_motif["letter_prob_matrix"]) / np.sum(np.exp(current_complementary_motif["letter_prob_matrix"]), axis=1, keepdims=True)
                #nucleotide_prob = np.ones((current_motif["width"], 4)) * 0.25
                s,d = scoreDist(current_complementary_motif["letter_prob_matrix"], nucleotide_prob=nucleotide_prob, gran=None, size=1000)
                cdf = np.cumsum(d)
                params, _ = curve_fit(logit_func, s, cdf, maxfev=1000) # obtain the correction coefficients and bias by curve fitting
                w, b = params
                current_complementary_motif["letter_prob_matrix"] *= w
                current_complementary_motif["bias"]               = b
            else:
                print('Unsupported normlization methods, return raw matrix')

            # center-aligned padding
            if current_motif["width"] < max_length:
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

    # stack the numpy matrices
    motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in motifs], axis=0)
    motifs_bias   = np.array([motif["bias"] for motif in motifs])
    complementary_motifs_matrix = np.stack([motif["letter_prob_matrix"] for motif in complementary_motifs], axis=0)
    complementary_motifs_bias   = np.array([motif["bias"] for motif in complementary_motifs])
    motif_names = [motif["name"] for motif in motifs]

    motifs_matrix = motifs_matrix.transpose(0, 2, 1)
    complementary_motifs_matrix = complementary_motifs_matrix.transpose(0, 2, 1)

    return (motifs_matrix,motifs_bias), (complementary_motifs_matrix,complementary_motifs_bias), motif_names, motif_length

class CenteredMaxPool1D(nn.Module):
    """Max-pooling that keeps the output aligned with the input positions.

    ``nn.MaxPool1d`` cannot express "same" padding, so a centred window would
    shift the output by half a kernel.  This module pads the sequence
    symmetrically before pooling, so output position ``i`` summarises input
    positions ``[i - k//2, i + k//2]`` and the length is preserved when
    ``stride=1``.

    Used wherever a per-position peak is needed while staying on the nucleotide
    grid -- for example reducing a motif activation map to one value per base.

    Parameters
    ----------
    kernel_size : int
        Odd window width.
    stride : int, default 1
        Pooling stride.
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        """Initialise and validate the window width.

        Parameters
        ----------
        kernel_size : int
            Must be odd, otherwise the window has no unique centre and the
            symmetric padding would shift the output by half a position.
        stride : int, default 1
            Pooling stride.

        Notes
        -----
        The assertion message is still in Chinese ("kernel_size must be an odd
        number"); it is a runtime string rather than a comment, so it was left
        untouched to avoid changing observable behaviour.
        """
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self, x):
        """Centre-padded max-pool over the length axis.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[batch, channel, length]``.

        Returns
        -------
        torch.Tensor
            Pooled tensor, ``[batch, channel, out_length]`` with
            ``out_length = floor(length / stride)`` for ``stride=1``.

        Notes
        -----
        Padding uses ``mode='replicate'``, which repeats the first and last
        element instead of introducing zeros.  That matters here: with
        zero-padding a max-pool could pick up a spurious 0 at the sequence edges,
        whereas replicating the edge value keeps the pooled signal within the
        range of the real activations.
        """
        # x shape: (batch, channel, length)
        padded_x = F.pad(x, (self.padding, self.padding), mode='replicate')
        pooled = F.max_pool1d(padded_x, kernel_size=self.kernel_size, stride=self.stride)
        return pooled