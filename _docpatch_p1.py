BLOCKS = []


def D(f, ln, txt):
    BLOCKS.append((f, ln, txt))


M = 'D:/GitHub/eScreen/escreen/motif_tool.py'
D(M, 1, '''"""Motif (PWM) utilities for the eScreen sequence-prior front end.

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
"""''')
D(M, 1, '')

D(M, 7, '''    """Parse a MEME-format motif file into a stacked probability matrix.

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
    """''')

D(M, 12, '''    # Header convention: line index 2 (0-based) of a MEME file is
    # 'ALPHABET= ACGT'.  The value is parsed but never used - the column order is
    # assumed to be A, C, G, T throughout this module.
''')
D(M, 13, '''    # Skip the fixed nine-line header (version, blank, ALPHABET, blank,
    # strands, blank, two background lines, blank) so that the loop starts at the
    # first MOTIF record.  Any extra header line shifts every offset below and
    # silently corrupts the parse instead of raising.
''')
D(M, 27, '''            # NOTE (documented, not fixed): this divides the PWM by its WIDTH,
            # not by a per-position normaliser, so the stored values are p / w
            # and the scale depends on the motif length.
''')
D(M, 28, '''            # Pad short motifs at the tail so that all PWMs share a height of
            # 29 bp and can be stacked.  Motifs of 29 bp or more are left
            # untouched (and would break the np.stack below if wider than 29).
''')

D(M, 45, '''    """Subtract each motif's per-position mean score (mean centring).

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
    """''')

D(M, 50, '''    """Left-compact the non-zero entries of each row, zero-filling the tail.

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
    """''')

D(M, 59, '''    """Load PWMs as a Conv1d kernel, with reverse complements appended.

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
    """''')

D(M, 60, '''    # Reverse complement: reverse the position axis and complement the
    # nucleotide axis (A,C,G,T -> T,G,C,A).  This is the step that turns the
    # 637 forward PWMs into the 1,274 filters of Supplementary Methods 2.3.6.
''')
D(M, 64, '''    # Channel order matters downstream: filters 0..n-1 are forward-strand and
    # filters n..2n-1 are reverse-strand; motif_names covers only the forward
    # half, so index i and i + n name the same motif on opposite strands.
''')
D(M, 66, '''    # Mean-centre (not z-score) so that a uniform background scores 0.
''')
D(M, 68, '''    # (n_motifs, L, 4) -> (n_filters, 4, L): the nn.Conv1d.weight layout, with
    # in_channels = 4 (A, C, G, T) and kernel_size = L.
''')

D(M, 73, '''    """Load PWMs as separate forward/reverse arrays with log-odds shifting and padding.

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
    """''')

D(M, 107, '''            # Log-odds-style shift: subtracting the uniform background (0.25)
            # makes a base at background frequency score 0, then the per-position
            # maximum is scaled to 1.
''')
D(M, 115, '''            # Tail padding: zeros are appended after the last motif position,
            # so the motif sits at the 5' end of the kernel.  max_length must be
            # at least the widest motif (28 for the shipped file, 35 in the
            # paper's configuration).
''')
D(M, 133, '''    # (n_motifs, L, 4) -> (n_motifs, 4, L): Conv1d weight layout.
''')

D(M, 142, '''    """Same as :func:`load_pwm_from_meme_c` but with *symmetric* padding.

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
    """''')

D(M, 187, '''            # Symmetric (centred) padding, unlike load_pwm_from_meme_c which
            # pads only at the tail.  When pad_total is odd the extra zero goes
            # after the motif (pad_bottom = pad_total - pad_top).
''')

D(M, 215, '''    """Odd-kernel 1-D max-pool that preserves length and motif centres.

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
    """''')
D(M, 216, '''        """Store the pooling geometry and derive the symmetric padding width.

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
        """''')
D(M, 217, '''        # Odd kernels have a well-defined centre, which is what keeps the pooled
        # output aligned with the input; an even kernel would shift the signal by
        # half a base per layer.
''')
D(M, 223, '''        """Apply centred max-pooling.

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
        """''')
D(M, 223, '''        # 'replicate' (edge) padding rather than zeros: with zero padding, a
        # window whose real activations are all negative would be dominated by
        # the padded zeros at the borders.
''')

A = 'D:/GitHub/eScreen/escreen/miaomiao_tool.py'
D(A, 1, '''"""Numeric helpers and epigenomic-signal utilities for the eScreen-local pipeline.

The module name ("miaomiao") is a lab nickname; functionally it is the
miscellaneous toolbox behind the local, 30-160 kb dissection model and the
figure post-processing.  It provides three groups of functionality:

1. **Summit / signal plumbing for the eScreen-local graph**
   (Supplementary Methods section 2.4.2).  ``match_summit`` assigns each element
   to its nearest epigenomic summit and records the signal there;
   ``capture_summit`` builds a summit-normalised local signal profile per 200 bp
   bin; ``get_value`` attaches raw BigWig profiles to an interval table with a
   per-row sanity flag.  These supply the peak nodes and the distance-decay edge
   weights (DNase, H3K27ac, CTCF, H3K27me3) of the bipartite graph.
2. **Score normalisation.**  ``sigmoid_normalization_point`` implements the
   two-anchor sigmoid mapping of raw LFC values onto 0-1
   (Supplementary Methods section 1.3); ``nonzero_mean`` implements the
   "ignore empty bins" averaging of the 200 bp bin scores (same section);
   ``norml`` and ``power_norm`` provide bounded display scales.
3. **Small safe-math helpers.**  ``safe_divide`` and ``safe_log`` avoid
   ``inf``/``nan`` from zero baselines, ``ln``/``lg`` are explicit-base logs,
   and ``remove_overlap`` performs an exact-row anti-join.

Conventions
-----------
* Genomic coordinates are 0-based, half-open ``[start, end)``, matching pyBigWig
  and BED.  No function here converts to 1-based coordinates.
* Bins are 200 bp wide: position ``p`` lies in bin ``floor(p / 200)`` and bin
  ``d`` covers ``[200 * d, 200 * d + 200)``.
* Most functions mutate the input DataFrame in place and rely on a unique,
  integer-like index (they address rows with ``df.loc[i, ...]``).

Caveats
-------
* :func:`capture_summit` calls ``catch_epi`` (defined in
  :mod:`escreen.genome_tool`) without importing it, and passes a two-element
  locus tuple where that helper expects three elements; see its Notes.
* Sentinel values (``1e5``, ``1e6``, ``1e8``) are used instead of ``NaN`` in two
  places, so they will be averaged unless filtered by group first.
* Nothing here imports the model, so the module can be used in the analysis
  notebooks without a GPU.
"""''')
D(A, 1, '')

D(A, 7, '''    """Assign each locus to its nearest epigenomic summit within a distance cut-off.

    Parameters
    ----------
    data : pandas.DataFrame
        Rows with ``chr`` and ``loci`` (a single genomic coordinate: the element
        centre).  Mutated in place with three new columns.
    summit_ref : pandas.DataFrame
        Summit table with ``chr``, ``start`` (summit coordinate), ``value``
        (signal) and ``peak_name``.
    threshold : int, default 200
        Maximum accepted distance in bp between ``loci`` and the nearest summit.

    Returns
    -------
    pandas.DataFrame
        ``data`` with:
        ``summit_group`` - ``peak_name`` of the nearest summit, or ``'loss'``;
        ``summit_value`` - that summit's ``value``, or the sentinel ``1e5``;
        ``summit_delta`` - the absolute distance used, or ``1e8`` when the
        chromosome carries no summit at all.

    Notes
    -----
    * The distance is unsigned (``abs``), so an element 100 bp upstream and one
      100 bp downstream of the same summit are treated identically.
    * ``summit_ref[summit_ref['chr'] == chromo]`` is a slice and is then extended
      with ``ref_sub['delta'] = ...``; under pandas copy-on-write this assignment
      may not persist and/or emits a ``SettingWithCopyWarning``.
    * The sentinels are large but finite, so downstream means/plots will include
      them unless rows with ``summit_group == 'loss'`` are dropped first.
    * Progress is printed with ``\\r`` on a single line.
    """''')

D(A, 10, '''        # Unsigned distance from the element centre to each summit on the same
        # chromosome; the nearest one wins regardless of direction.
''')
D(A, 15, '''            # Beyond the cut-off: mark the element as 'loss' and use large
            # sentinels so these rows sort last rather than acting as if they had
            # a real zero-distance summit hit.
''')

D(A, 32, '''    """Store a summit-normalised local signal profile for every bin.

    For each bin this samples the track at ``sequence_len - kmer + 1`` offsets
    starting at the bin's 5' end and divides every sample by the height of the
    nearest summit, producing the shape of the local signal around an annotated
    peak rather than its absolute magnitude.

    Parameters
    ----------
    data : pandas.DataFrame
        Bins with ``chr``, ``start``, ``end``.  Mutated in place.
    summit_ref : dict of pandas.DataFrame
        Mapping ``chrom -> summit table`` with ``start`` and ``value`` columns.
    bw_dict : dict
        Mapping mark name -> open ``pyBigWig`` handle.
    epi : str, default ''
        Mark name; used both as the ``bw_dict`` key and as the prefix of the two
        generated columns.
    sequence_len : int, default 200
        Length of the per-bin window in bp (one 200 bp bin).
    kmer : int, default 40
        Offset used when stepping through the window; the loop therefore runs
        ``sequence_len - kmer + 1`` = 161 times for the defaults.

    Returns
    -------
    pandas.DataFrame
        ``data`` with ``'{epi}_nearest_summit'`` (the normaliser: summit signal,
        ``1e6`` where unknown) and ``'{epi}_summit_weight'`` (object column
        holding one float array per row).

    Notes
    -----
    * ``summit_ref`` is keyed by chromosome here, whereas :func:`match_summit`
      takes a flat frame - the same idea needs two different input shapes.
    * Only bins whose nearest summit is closer than 500 bp get a real
      normaliser; the rest keep the ``1e6`` value from ``data.fillna(1e6)``,
      which drives their weights to approximately zero without dividing by zero.
    * ``catch_epi`` is neither imported in this module nor called with the
      3-element ``(chrom, start, end)`` tuple it expects, so as written this
      raises ``NameError`` (or ``IndexError`` if the symbol is injected).  The
      intended semantics are the per-offset ratio documented above; the code is
      left unchanged deliberately.
    * ``kmer`` is only an offset, not a motif length: single bases are sampled,
      not k-mers.
    """''')

D(A, 37, '''                # Distance to the bin CENTRE, not the bin start; both bins and
                # summits are 0-based half-open coordinates.
''')
D(A, 39, '''                # Only summits within 500 bp of the bin centre are recorded;
                # all other bins keep the 1e6 fill applied below.
''')
D(A, 45, '''    # Unmatched bins get 1e6, so dividing by this sentinel drives the weight to
    # ~0 instead of raising a division by zero.
''')
D(A, 49, '''        # One ratio per offset k in [0, sequence_len - kmer]:
        # signal(start + k) / summit height, i.e. a summit-normalised local
        # profile.  np.newaxis adds the channel axis, squeezed away immediately.
''')

D(A, 58, '''    """Attach raw BigWig profiles to an interval table with a per-row sanity flag.

    Parameters
    ----------
    data : pandas.DataFrame or None
        Intervals with ``chr``, ``start``, ``end``.  ``None`` passes straight
        through unchanged.
    bw : pyBigWig.bigWig
        Open track handle.
    feature : str, default ``'profile'``
        Name of the output column that receives the per-base arrays.
    threshold : float, default 0.5
        Fraction of the interval that one repeated value may occupy before the
        row is considered degenerate.
    _filter_ : bool, default True
        Drop rows whose ``ok`` flag is False.
    _verbose_ : bool, default True
        Print the number of rows with ``y > 0`` and ``y == 0`` after filtering.

    Returns
    -------
    pandas.DataFrame
        ``data`` with ``feature`` (object dtype, one array per row) and ``ok``
        (bool); optionally row-filtered.

    Notes
    -----
    * Coordinates are sorted with ``min``/``max`` before querying, so inverted
      intervals still work.  ``NaN`` (BigWig gaps) become 0.
    * The ``ok`` expression
      ``(most_common_count <= (end - start) * threshold) | (most_common_count != 0)``
      is True for any non-empty selection, because a repeat count is only 0 for
      an empty array.  In practice ``_filter_`` therefore removes only the rows
      whose BigWig query raised - those are filled with ``np.zeros(200)``
      regardless of the true interval length.
    * The verbose branch dereferences ``data.y``, so it raises ``KeyError`` when
      the frame has no ``y`` label column.
    """''')

D(A, 70, '''                # Degeneracy check: how much of the interval is occupied by a
                # single repeated value (an all-zero stretch, a masked region, ...).
''')
D(A, 72, '''                # NOTE (documented, not fixed): the second disjunct makes this
                # True for every non-empty selection, so 'ok' effectively only
                # marks rows whose BigWig query failed.
''')
D(A, 74, '''                # Failed query: substitute a 200-bp zero vector regardless of
                # the real interval length, and flag the row as not ok.
''')

D(A, 87, '''    """Natural logarithm: ``log(x) / log(e)``."""''')
D(A, 90, '''    """Base-10 logarithm via the change-of-base formula: ``log(x) / log(10)``."""''')

D(A, 94, '''    """Fit a sigmoid through two anchor points and apply it to an array.

    This is the sigmoid normalisation of Supplementary Methods section 1.3, which
    maps raw gRNA LFC values onto a unified 0-1 scale so that scores are
    comparable across readout types and selection directions.  The paper fixes
    the two parameters empirically; here they are *solved* from two anchor pairs.

    Parameters
    ----------
    array : array_like
        Raw scores (typically LFCs), any shape.
    s1 : tuple of float, default ``(0.0, 0.3)``
        ``(x1, y1)``: input value mapped exactly to output ``y1``.
    s2 : tuple of float, default ``(-0.3, 0.7)``
        ``(x2, y2)``: input value mapped exactly to output ``y2``.

    Returns
    -------
    numpy.ndarray
        ``1 / (1 + exp(-n * array + b))``, same shape as ``array``, values in
        (0, 1).

    Notes
    -----
    * ``n`` and ``b`` are printed, not returned, so the calibration cannot be
      reused by the caller without copying the printout or refitting.
    * The two anchors must differ in ``x`` and satisfy ``0 < y < 1``; ``y`` equal
      to 0 or 1 makes the logit infinite and the fit degenerate.
    * ``e`` is the literal ``2.71828`` rather than ``np.e``; the relative error
      (~1e-6) is irrelevant for scores of order 1.
    * With the defaults, ``LFC = 0`` maps to 0.3 and ``LFC = -0.3`` to 0.7, i.e.
      the mapping is calibrated for a negative-selection screen in which more
      negative LFC means more active.  For positive selection, pass anchors with
      the sign of ``x`` flipped.
    """''')
D(A, 96, '''    # Solve y = 1 / (1 + exp(-n*x + b)) through the two anchors, i.e. the logit
    # of the anchor outputs is linear in x:
    #     logit(y1) = -n*x1 + b        logit(y2) = -n*x2 + b
    # hence n = (logit(y1) - logit(y2)) / (x2 - x1) and b from either anchor.
''')
D(A, 104, '''    # Apply with the same sign convention used when solving: exp(-n*x + b).
''')

D(A, 110, '''    """Mean over an axis that ignores zeros (zeros are missing, not measurements).

    Parameters
    ----------
    arr : numpy.ndarray
        Numeric array.
    axis : int
        Axis to reduce.

    Returns
    -------
    numpy.ndarray
        Mean of the non-zero entries along ``axis``; ``NaN`` for a slice that is
        entirely zero (``np.nanmean`` of all-``NaN``).

    Notes
    -----
    This is the "exclude zero values, then average" rule for the 200 bp bin
    scores of Supplementary Methods section 1.3: a bin with no gRNA activity is
    missing data, not a zero-activity measurement.  Zeros are converted through a
    copy (``np.where``), so the input array is not modified.
    """''')

D(A, 117, '''    """Min-max normalise to [0, 1].

    Parameters
    ----------
    x : torch.Tensor or array_like
        Input.  Tensors are moved to CPU and converted with ``.numpy()`` first.

    Returns
    -------
    numpy.ndarray
        ``(x - min) / (max - min)`` with min/max taken over the **whole** array,
        not per row or per column.

    Notes
    -----
    A constant input produces ``0/0`` (runtime warning) and ``nan`` output; there
    is no guard.  The return type is always ``ndarray``, so tensor input loses its
    device and dtype.
    """''')

D(A, 126, '''    """Element-wise division in which a zero denominator yields 0.

    Parameters
    ----------
    a, b : array_like
        Numerator and denominator, broadcast to a common shape.

    Returns
    -------
    numpy.ndarray
        float64 result with 0.0 wherever ``b == 0``.

    Notes
    -----
    Needed for the per-bin effect sizes of Supplementary Methods section 2.4.5,
    where dividing by a zero baseline bin would otherwise produce ``inf`` and
    poison the downstream Pearson correlation / TVD.
    * Both inputs go through ``np.array(...)``, so the ``pd.Series`` branch at the
      end is unreachable and index alignment is lost; pass pre-aligned arrays.
    * Broadcast shapes other than element-wise are not supported: the mask and
      the assignment assume ``a`` and ``b`` share a shape.
    """''')

D(A, 136, '''    """Element-wise logarithm with 0 for non-positive inputs.

    Parameters
    ----------
    a : array_like
        Input values.
    base : float, default 2
        Logarithm base, applied with the change-of-base formula.

    Returns
    -------
    numpy.ndarray
        float64 result; entries ``<= 0`` map to 0.0 instead of ``-inf``/``nan``.

    Notes
    -----
    Used for fold-change style summaries where a zero count must not produce
    ``-inf``.  As in :func:`safe_divide`, the ``pd.Series`` branch is unreachable
    because the input is converted to ``ndarray`` first.
    """''')

D(A, 145, '''    """Anti-join two frames on exact row identity.

    Parameters
    ----------
    df1, df2 : pandas.DataFrame
        Frames to compare; they should share column names and comparable dtypes.

    Returns
    -------
    df1_non_overlap, df2_non_overlap : pandas.DataFrame
        The rows of each frame whose full tuple of column values does not appear
        in the other frame.

    Notes
    -----
    * "Overlap" here means *identical row content*, not genomic interval overlap.
      For coordinate overlap use :func:`escreen.genome_tool.ref_filter` or
      bedtools.
    * The inner merge is on all shared columns, so column order and dtypes must
      match; reordered or retyped columns are treated as non-overlapping.
    * ``df.apply(tuple, 1)`` materialises one Python tuple per row - fine for
      thousands of rows, slow for millions.
    """''')

D(A, 152, '''    """Gamma-compress absolute values onto [0, 1].

    Parameters
    ----------
    data : array_like
        Values of any sign; the sign is discarded.
    max_val : float, default 10
        Magnitude mapped to 1.0.
    gamma : float, default 0.424
        Exponent; ``gamma < 1`` expands small magnitudes, ``gamma > 1``
        compresses them.

    Returns
    -------
    numpy.ndarray
        ``clip((|data| / max_val) ** gamma, 0, 1)``.

    Notes
    -----
    The clip is what bounds the scale: magnitudes above ``max_val`` saturate at 1
    and become indistinguishable.  Because the absolute value is used, this is
    only suitable for magnitude colour scales, not for signed heatmaps.
    """''')
D(A, 154, '''    # Clip bounds the scale; |data| discards sign.
''')
