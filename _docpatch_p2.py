BLOCKS = []


def D(f, ln, txt):
    BLOCKS.append((f, ln, txt))


G = 'D:/GitHub/eScreen/escreen/genome_tool.py'
D(G, 1, '''"""Genomic interval, sequence and signal-track helpers shared by the eScreen analyses.

This is the plumbing layer of eScreen: it turns genomic coordinates into
one-hot DNA, reads BigWig signal over intervals, converts intervals between
genome builds, tiles and merges BED intervals, and draws the locus-level panels
used in the figures.  Specifically it implements:

* sequence retrieval and one-hot encoding as in Supplementary Methods section
  2.3.1 - hg38 sequence via pyfaidx (0-based half-open slicing), A/C/G/T
  channels and an all-zero ``N``;
* the hg19 -> hg38 LiftOver plus zero-gap interval merging used to build the
  K562 enhancer cluster set of Supplementary Methods section 2.4.3
  (:func:`convert_genome`, :func:`merge_intervals`);
* the eQTL and GWAS variant counting behind the CRE annotations of
  Supplementary Methods sections 1.6.2 and 1.6.3
  (:func:`eqtlboxplot_draw`, :func:`gwashistplot_draw`);
* epigenomic track retrieval (DNase, ATAC, H3K27ac, H3K4me3, CTCF, H3K27me3)
  for the peak nodes of the eScreen-local bipartite graph
  (Supplementary Methods section 2.4.2) and for the genome-browser style panels
  (:func:`catch_epi`, :func:`draw_loci_context`, :func:`draw_loci_context_ax`).

Coordinate conventions
----------------------
* All coordinates are 0-based, half-open ``[start, end)`` (BED convention),
  which is also what pyfaidx slicing and ``pyBigWig.values()`` expect.  Nothing
  in this module converts to or from 1-based coordinates; the only place a
  1-based source appears is the eQTL table, and the counting there compensates
  only by using inclusive comparisons on both ends.
* ``chr_list`` covers chr1-chr22 and chrX.  chrY, chrM and alt/random/decoy
  contigs are dropped by :func:`genome_sequence` and friends.
* Interval frames must carry ``chr``, ``start``, ``end`` columns and a unique
  integer-like index: several helpers address rows by label
  (``df.loc[i, ...]``), so duplicate index values silently address the wrong
  row.

Portability caveats
-------------------
Several helpers were written for the authors' cluster and are not
self-contained:

* :func:`draw_loci_context`, :func:`draw_loci_context_ax` and
  :func:`eqtlboxplot_draw` contain hard-coded absolute paths such as
  ``/cluster2/huanglab/liquan/data/K562/DNase.bigWig``;
* :func:`merge_intervals` references ``BedTool``,
  :func:`genome_sequence_withoutcoding` references ``rainbow_tqdm``, and
  :func:`eqtlboxplot_draw`` references ``time`` - none of these three names is
  imported in this module, so those functions raise ``NameError`` unless the
  caller injects the symbols into the module namespace.

These defects are documented rather than repaired, because the module's runtime
behaviour (and every figure already produced from it) must stay unchanged.
"""''')
D(G, 1, '')

D(G, 14, '''# Nucleotide -> one-hot channel.  ``N`` maps to all zeros (a fully "off"
# position) rather than to a fifth channel - the same convention the model uses
# for masked/padding bases and for the all-zero IG baseline.  Any other character
# (lower case, IUPAC ambiguity codes, gaps) makes one_hot() raise KeyError.
''')
D(G, 15, '''# Canonical chromosomes only: chrY, chrM and alt/random/decoy contigs are absent,
# so every helper that filters on this list silently drops rows on those contigs.
''')
D(G, 16, '''# Complement table used by reverse_seq(); N is self-complementary.
''')

D(G, 19, '''    """Complement a nucleotide string in place order (not a reverse complement).

    Parameters
    ----------
    seq : str
        Upper-case nucleotide string over A/C/G/T/N.

    Returns
    -------
    str
        Same length, each character replaced through ``trans_dict``
        (A<->T, C<->G, N->N).

    Raises
    ------
    KeyError
        For any character outside A/C/G/T/N: lower-case input, IUPAC codes,
        gaps, whitespace.

    Notes
    -----
    Despite the name, only the complement is taken.  Callers that need the
    reverse complement must also reverse the string
    (``reverse_seq(seq)[::-1]``, as :func:`find_contextual` does).
    """''')

D(G, 22, '''    """One-hot encode a nucleotide string.

    Parameters
    ----------
    seq : str
        Upper-case sequence over A/C/G/T/N.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(seq), 4)``, integer dtype (stacked from Python lists, so
        ``int64`` on 64-bit builds).  Channel order is A, C, G, T.

    Notes
    -----
    ``N`` becomes ``[0, 0, 0, 0]``, so a run of Ns is indistinguishable from
    padding; this is the same representation the model sees for masked bases.
    There is no upper-casing here - callers must pass upper-case input
    (:func:`genome_sequence` calls ``.upper()`` first).
    """''')

D(G, 25, '''    """Read a BigWig track over a locus, zero-padding anything off the chromosome.

    Parameters
    ----------
    bw : pyBigWig.bigWig or None
        Open handle.  ``None`` short-circuits to ``np.zeros(1)``.
    loci : sequence
        ``(chrom, start, end)`` with ``chrom`` a ``str`` such as ``'chr8'`` and
        0-based half-open coordinates that may be negative or beyond the
        chromosome end.

    Returns
    -------
    numpy.ndarray
        Per-base signal of length ``end - start``; out-of-chromosome parts are
        filled with zeros so the length always matches the requested span.

    Notes
    -----
    * The first query is attempted directly; pyBigWig raises when the interval
      leaves the chromosome and the ``except`` branch rebuilds the vector from
      the clipped part plus zeros.  ``bound = int(bw[chrom])`` is the chromosome
      length.
    * The three clipping branches cover ``start < 0 < end <= bound``,
      ``0 <= start < bound < end`` and ``start < 0 < bound < end``.  If ``start``
      itself is beyond ``bound`` none of them applies and the second
      ``bw.values`` call raises again, so callers must keep ``start`` inside the
      chromosome.
    * A missing handle or an empty ``loci`` list returns ``np.zeros(1)``: a
      length-1 placeholder with the wrong shape for most uses.  Callers check
      length, never shape.
    """''')

D(G, 34, '''    # Direct query: valid when the interval lies fully inside the chromosome.
''')
D(G, 36, '''        # Rebuild the out-of-range part from zeros so that the returned length
        # is always end - start; 'bound' is the chromosome length.
''')

D(G, 52, '''    """Extract hg38 sequence for every interval and one-hot encode it.

    Implements the sequence step of Supplementary Methods section 2.3.1: fetch
    the reference sequence of each cCRE-centred window and one-hot encode it.

    Parameters
    ----------
    data : pandas.DataFrame
        Interval table with ``chr``, ``start``, ``end`` and a unique index.
    fa : pyfaidx.Fasta
        Indexed hg38 FASTA.  Slicing is 0-based half-open, matching BED.
    chr_list : list of str, default ``chr_list``
        Chromosomes to keep.  Rows on other contigs are dropped *before*
        indexing, so the result can be shorter than ``len(data)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_kept, L, 4)`` where ``L = end - start``; every interval must
        have the same length or ``np.stack`` raises.  Row order follows
        ``data.index``.

    Notes
    -----
    * Rows are addressed with ``data.loc[i, ...]`` (label based), so the index
      must be unique.
    * ``.upper()`` hard-masks soft-masked (lower-case) reference bases; ``N``
      runs stay all-zero channels.
    * A ``tqdm`` progress bar is shown; use
      :func:`_genome_sequence_withoutcoding_` for the un-encoded variant.
    """''')
D(G, 53, '''    # Drop non-canonical contigs first so every kept row can be sliced from the
    # FASTA; the returned array follows data.index and may be shorter than
    # len(data).
''')
D(G, 55, '''        # pyfaidx slicing is 0-based half-open (BED convention), and .upper()
        # converts soft-masked bases to hard ACGT.
''')

D(G, 61, '''    """Extract raw interval sequences, without one-hot encoding.

    Parameters
    ----------
    data : pandas.DataFrame
        Interval table with ``chr``, ``start``, ``end``.
    fa : pyfaidx.Fasta
        Indexed genome FASTA.
    chr_list : list of str, default ``chr_list``
        Chromosomes to keep.

    Returns
    -------
    list of str
        One sequence per kept row, in index order.  Case is preserved (no
        ``.upper()``), so soft-masked bases survive.

    Notes
    -----
    The underscore-suffixed name marks this as the internal fast path: no
    progress bar, no encoding, no length check.  The caller is responsible for
    uniform lengths if the result is to be stacked.
    """''')

D(G, 66, '''    """Same as :func:`_genome_sequence_withoutcoding_`, with a progress bar.

    Parameters
    ----------
    data : pandas.DataFrame
        Interval table with ``chr``, ``start``, ``end``.
    fa : pyfaidx.Fasta
        Indexed genome FASTA.
    chr_list : list of str, default ``chr_list``
        Chromosomes to keep.

    Returns
    -------
    list of str
        One sequence per kept row, in index order, case preserved.

    Raises
    ------
    NameError
        ``rainbow_tqdm`` is used but never imported in this module, so the call
        fails on entry unless the symbol is injected by the caller.
    """''')

D(G, 71, '''    """Tile intervals into fixed-width bins of ``length`` bp.

    Parameters
    ----------
    df : pandas.DataFrame
        Interval table with ``chr``, ``start``, ``end``.
    length : int, default 200
        Bin width in bp.  The paper discretises CREs and windows into 200 bp
        bins (Supplementary Methods section 2.4.5), so bin index =
        ``floor(position / 200)`` and bin ``d`` covers ``[200*d, 200*d + 200)``.

    Returns
    -------
    pandas.DataFrame
        Columns ``chr``, ``start``, ``end``, one row per emitted bin.  The bin
        count is also printed.

    Notes
    -----
    * While more than ``length`` bp remain, ``[start, start + length)`` is
      emitted, so every non-final bin is exactly ``length`` wide.
    * The final bin uses ``max(start + length, end)``, which **extends a short
      remainder past the original end** to fill a whole bin instead of emitting
      a narrow last bin.  Last-bin coordinates may therefore exceed the input
      interval; :func:`catch_epi` then zero-pads if the overhang leaves the
      chromosome, and pyfaidx truncates at the contig end.
    * ``start == end`` emits nothing; ``start > end`` is not checked and yields an
      empty range.
    """''')
D(G, 78, '''            # Full-width bin: emit [start, start + length) and advance.
''')
D(G, 83, '''            # Final, short bin: padded up to a full 'length' bp (it may overshoot
            # the original end) so every bin has the same width.
''')

D(G, 91, '''    """Sort and zero-gap merge BED intervals with bedtools, after light repair.

    This is the merging rule of Supplementary Methods section 2.4.3: intervals
    that overlap **or are directly adjacent** are collapsed into one continuous
    region, so the output has no bookended neighbours.

    Parameters
    ----------
    df : pandas.DataFrame
        Interval table with ``chr``, ``start``, ``end``.

    Returns
    -------
    pandas.DataFrame
        Columns ``chr``, ``start``, ``end``: the merged, chromosome-sorted
        interval set.

    Notes
    -----
    * Two repairs run before merging: inverted intervals (``start > end``) are
      swapped, and intervals shorter than 3 bp are widened to ``start + 20``
      (the literal constant is 20, not 3).
    * ``bedtools merge`` with default options is zero-gap, so ``[0, 100)`` and
      ``[100, 200)`` become ``[0, 200)``; pass ``-d`` to allow a gap.
    * Sorting is ``bedtools sort`` (string order: chr1, chr10, chr11, ..., chr2),
      so do not assume numeric chromosome order downstream.
    * Two temporary files, ``intervals.bed`` and ``merged_intervals.bed``, are
      written to and removed from the **current working directory**; concurrent
      calls in one directory clobber each other.
    * Raises ``NameError`` because ``BedTool`` is not imported in this module.
    """''')
D(G, 95, '''        # Repair inverted intervals produced by upstream parsing.
''')
D(G, 99, '''        # bedtools merge drops zero-length intervals, so anything under 3 bp is
        # widened to 20 bp first (20 is literal here, unrelated to the 3 bp test).
''')
D(G, 102, '''    # sort | merge with bedtools defaults: overlapping AND bookended intervals
    # are collapsed (zero-gap merge, Supplementary Methods section 2.4.3).
    # Temporary files land in the current working directory.
''')

D(G, 114, '''    """Lift one interval between genome builds, nudging the start when needed.

    Implements the hg19 -> hg38 conversion used for the dbSUPER intervals of
    Supplementary Methods section 2.4.3 and for the eQTL/GWAS comparisons.

    Parameters
    ----------
    region : sequence
        ``(chrom, start, end)`` in the **source** build, 0-based half-open.
    lo : pyliftover.LiftOver or None
        Chain-file converter.  With ``None`` the body is skipped and the function
        returns ``None``.

    Returns
    -------
    tuple or None
        ``(chrom, start, end)`` in the target build, with the original interval
        length preserved.  On failure, ``('0', 0, 0)`` after printing the locus.

    Notes
    -----
    * Only the start is converted; the end is reconstructed as ``start + d``
      where ``d`` is the original length, so a partially mapped interval keeps its
      length rather than its original end.
    * LiftOver returns ``None`` for unmappable loci and an empty list for some
      gaps; both paths shift the whole interval one base at a time (``a += 1;
      b += 1``) for at most 100 attempts, then give up with ``('0', 0, 0)``.
      At most 100 bp of leading unmappable sequence can therefore be skipped.
    * ``converted[0]`` is trusted without checking that the returned end is
      greater than the start; LiftOver can return a reversed pair for intervals
      straddling an assembly gap.
    * ``convert_coordinate`` expects 0-based starts; a 1-based start would shift
      every converted interval by one base.
    """''')
D(G, 116, '''        # d is the interval length preserved across builds; only the start is
        # looked up, so a straddling interval keeps its length, not its end.
''')
D(G, 126, '''                    # Unmappable here: shift the whole interval one base
                    # downstream and retry (capped at 100 attempts below).
''')

D(G, 142, '''    """Keep the reference intervals that overlap at least one query interval.

    Parameters
    ----------
    r : pandas.DataFrame
        Reference table with ``chr``, ``start``, ``end`` (e.g. candidate CREs).
    d : pandas.DataFrame
        Query table with ``chr``, ``start``, ``end`` (e.g. gRNAs or 200 bp
        bins).

    Returns
    -------
    outref : pandas.DataFrame
        Subset of ``r`` restricted to chromosomes that had at least one hit.
    df : pandas.DataFrame
        Concatenation of every query row overlapping any reference row.

    Notes
    -----
    * Overlap is the union of three cases (query start inside, query end inside,
      reference fully inside the query) on integer-cast coordinates.  Intervals
      that merely touch (``query.end == ref.start``) count as overlapping.
    * ``outref`` is built with ``r.loc[r['chr'] == chromo]``, i.e. **all**
      reference rows on a chromosome that had any hit, appended once per matched
      reference row.  It is not synchronised with ``df`` and may contain
      duplicates; callers needing a one-to-one mapping must re-merge on
      coordinates.
    * With no overlap, ``df`` is an empty DataFrame and ``'No activity guide!'``
      is printed.
    """''')
D(G, 150, '''            # Three overlap cases: query start inside, query end inside, or the
            # reference interval fully contained in the query.
''')

D(G, 168, '''    """Split a ``Coordinates`` string into ``chr``/``start``/``end``/``strand``.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a ``Coordinates`` column formatted as
        ``'...|chr:start-end:strand'``; the last ``|``-separated field is used.
    col : str, default ``'Coordinates'``
        Name of the coordinate column.

    Returns
    -------
    pandas.DataFrame
        **Only** ``['chr', 'start', 'end', 'strand']``.  All other columns are
        dropped from the returned frame, although ``df`` itself is mutated in
        place with the three new coordinate columns.

    Notes
    -----
    * Missing start/end values become 0; a missing strand segment becomes ``'*'``
      and a message is printed.
    * ``start``/``end`` are parsed straight from the string with ``astype(int)``,
      so the output inherits the source convention - no 0-/1-based conversion is
      performed anywhere.
    * A value with no ``':'`` yields ``NaN`` start/end, which ``fillna(0)`` turns
      into 0 rather than raising.
    """''')
D(G, 170, '''    # '<tag>|chr:start-end:strand' -> take the last pipe-separated field.
''')

D(G, 185, '''    """Locate a 20 bp sequence inside a flanked window and return its 30 bp context.

    Used to re-anchor a stored guide/CRE sequence to the genome before feature
    extraction: the last 20 bp of the row's sequence are searched inside
    ``[start - expand_flank, end + expand_flank]`` on both strands, and the
    oriented 30 bp neighbourhood is returned together with its genomic position.

    Parameters
    ----------
    data : pandas.DataFrame
        Rows with ``chr``, ``start``, ``end`` and ``sequence``.  Only the last
        20 bp of ``sequence`` are used (spaces stripped).
    fa : pyfaidx.Fasta or None
        Indexed genome FASTA; required in practice.
    expand_flank : int, default 30
        Half-width of the search window around ``[start, end]``, in bp.
    upstream : int, default 4
        Bases of context added 5' of the match.
    downstream : int, default 6
        Bases of context added 3' of the match.

    Returns
    -------
    direction : list of str
        ``'+'``, ``'-'``, or ``'-1'`` when the row cannot be placed.
    seq_adj : list of str
        Oriented context, always ``total_len`` (30) upper-case characters;
        ``'N' * total_len`` for unplaceable rows.
    position_adj : list of int
        0-based genomic start of the matched 20-mer on the forward strand, or
        ``-1`` for unplaceable rows.

    Notes
    -----
    * pyfaidx windows are 0-based half-open, so ``position_adj`` can be used
      directly as a BED start.
    * The 30 bp context is ``20 + upstream + downstream``: for a forward hit it
      starts ``upstream`` bp before the match, for a minus-strand hit the window
      is shifted by ``-downstream`` so the same 30 bp of oriented sequence is
      recovered after reverse-complementing.
    * Minus-strand hits return the reverse complement, so ``seq_adj`` always
      reads 5' -> 3' in the guide/gene orientation.
    * If neither strand matches, the fallback returns
      ``fa[chr][start - upstream : start + 20 + downstream]`` with
      ``direction='+'`` and reports ``start`` (not a match position), so
      ``position_adj`` is then the annotated start rather than a search result.
    * Rows on contigs outside ``chr_list`` are marked ``'-1'`` /
      ``'N' * total_len`` / ``-1`` without a FASTA lookup.
    * The ``except`` branch references an undefined ``s1``, so a row with a
      malformed coordinate raises ``NameError`` instead of being marked
      unplaceable.
    """''')
D(G, 191, '''        # Only the last 20 bp of the stored sequence serve as the search key;
        # spaces (alignment padding) are stripped first.
''')
D(G, 207, '''        # Search window: +/- expand_flank around the annotated interval,
        # 0-based half-open because pyfaidx is.
''')
D(G, 209, '''        # Forward-strand match first, then the reverse complement.
''')
D(G, 214, '''                # Neither strand matched: fall back to a fixed 30 bp window
                # starting 'upstream' bp before the annotated start.
''')

D(G, 229, '''    """Count eQTLs per group interval and draw the group-wise comparison panel.

    Implements the eQTL annotation of Supplementary Methods section 1.6.2
    (CRE-vs-eQTL overlap, counted per group) and renders the two-group box +
    strip panel with a p-value bracket.

    Parameters
    ----------
    data : list of pandas.DataFrame
        Groups to compare; each frame needs ``chr``, ``start``, ``end``.  Frames
        are mutated in place (``snp`` and ``label`` columns are added).
    chromo : str
        Unused; kept for call-site compatibility.
    lo : pyliftover.LiftOver or None, default None
        Optional converter applied to each interval before counting.
    label : str, default ``'eQTL boxplot figure'``
        x-axis label.
    label_list : list of str, default []
        One label per group, consumed positionally against ``data``.
    guide_expand : bool, default False
        Widen each interval by ``expand`` bp on both sides (guide footprint
        instead of the bare CRE).
    expand : int, default 90
        Expansion in bp used when ``guide_expand`` is True.
    sample_limit : int, default 200
        Unused.

    Returns
    -------
    None
        The figure is drawn with ``plt.show()``; no object is returned.

    Notes
    -----
    * The eQTL table is read from the hard-coded path
      ``/cluster2/huanglab/liquan/data/eQTL/ciseqtl-loci.txt`` with columns
      ``SNPChr`` (bare integer chromosome) and ``SNPPos``, so the function is not
      portable as shipped.
    * ``chrX`` rows get ``NaN`` because ``SNPChr`` is compared after
      ``int(chrom.split('chr')[1])``.
    * Counting is inclusive on both ends
      (``SNPPos >= start`` and ``SNPPos <= end``).  ``SNPPos`` is 1-based in the
      source while ``start``/``end`` are 0-based half-open, so a variant sitting
      exactly on ``start`` is counted in both neighbouring bins.
    * The Liftover branch compensates for reversed conversions (``s > e``) by
      shifting the original interval instead of trusting the converted pair;
      otherwise it takes the converted pair as-is.
    * Exactly two groups are assumed by the t-test and the bracket colouring;
      ``ttest_ind`` is applied with no normality/variance check and the p-value
      is annotated as ``p = ...``.
    * A falsy ``label`` triggers ``label = time.time()``, and ``time`` is not
      imported, so that branch raises ``NameError``.
    """''')
D(G, 232, '''    # Hard-coded eQTL table (not portable).  SNPChr is a bare integer (1..22) and
    # SNPPos is 1-based, which is why the counting test below is inclusive on both
    # ends.
''')
D(G, 244, '''            # chrX is skipped because SNPChr is matched via int(chrom[3:]).
''')
D(G, 256, '''                # Reversed conversion (s > e) means the interval straddled an
                # assembly gap; shift the original interval instead of trusting
                # the converted pair.
''')
D(G, 275, '''                    # Inclusive on both ends; see the coordinate-convention note
                    # at the top of this function.
''')

D(G, 343, '''    """Count GWAS variants per group interval and draw the group totals.

    Implements the GWAS annotation of Supplementary Methods section 1.6.3
    (CRE-vs-GWAS-locus overlap) and renders one bar per group with the summed
    variant count.

    Parameters
    ----------
    data : list of pandas.DataFrame
        Groups; each frame needs ``chr``, ``start``, ``end``.  Mutated in place
        with a ``gwas`` count column and a ``label`` column.
    chromo : str
        Unused; kept for call-site compatibility.
    gwas_data : pandas.DataFrame
        Variant table with ``chromosome`` (integer) and
        ``base_pair_location`` (1-based position) columns.
    lo : pyliftover.LiftOver or None, default None
        Optional converter applied to each interval before counting.
    label : str, default ``'GWAS figure'``
        x-axis label.
    label_list : list of str, default []
        One label per group, consumed positionally.
    guide_expand : bool, default False
        Widen each interval by ``expand`` bp on both sides.
    expand : int, default 90
        Expansion in bp used when ``guide_expand`` is True.

    Returns
    -------
    None
        The figure is drawn with ``plt.show()`` and the per-group totals are
        printed.

    Notes
    -----
    * The Liftover loop retries with ``start += 1; end += 1`` and has **no attempt
      cap**: a locus that keeps returning an empty list without ever returning
      ``None`` loops forever.  ``None`` exits via ``status = 2`` (counting 0).
    * ``chrX`` rows get ``NaN`` for the same integer-chromosome reason as
      :func:`eqtlboxplot_draw`.
    * ``start == end`` is recorded as 0 rather than queried, because an empty
      interval cannot contain a variant.
    * Variants are counted with inclusive bounds on ``base_pair_location``
      (1-based), so the same off-by-one caveat as in
      :func:`eqtlboxplot_draw` applies.
    * The bar heights are sums of counts, not densities, so groups covering more
      intervals automatically produce taller bars.
    """''')
D(G, 359, '''                # Keep nudging the start until the locus maps.  There is no
                # attempt cap here (unlike convert_genome): only a None return from
                # LiftOver breaks the loop, so an endlessly empty result hangs.
''')
D(G, 392, '''                    # GWAS summary-table columns: chromosome (integer) and
                    # base_pair_location (1-based).
''')
D(G, 415, '''    # One bar per group: total number of GWAS variants inside the group's
    # intervals (sum, not mean).
''')

D(G, 431, '''    """Plot six K562 epigenomic tracks as offset, max-normalised profiles.

    Parameters
    ----------
    loci : sequence
        ``(chrom, start, end)``; 0-based half-open, matching pyBigWig.

    Returns
    -------
    None
        Draws with ``plt.show()``.

    Notes
    -----
    * Tracks are hard-coded to ``/cluster2/huanglab/liquan/data/K562/*.bigWig``:
      DNase, ATAC, H3K27ac, H3K4me3, CTCF, H3K27me3 - the marks used for the
      peak nodes of the eScreen-local graph
      (Supplementary Methods section 2.4.2).
    * Each track is divided by its own maximum and shifted by a fixed offset
      (2.0, 1.5, 1.0, 0.5, 0.0, -0.5), so the y axis is a stack index, not a
      signal unit.  A flat zero track gives ``0 / 0 = nan`` and is drawn as a
      gap.
    * The x axis is one point per base over ``[start, end)``, so a multi-megabase
      window allocates one float per base per track.
    """''')
D(G, 433, '''    # Hard-coded K562 tracks from the lab cluster: the six marks behind the peak
    # nodes and the genome-context panels.
''')
D(G, 461, '''    # Max-normalise each track and add a fixed offset so six profiles share one
    # axis; the y value is a stack index, not a signal value.
''')

D(G, 480, '''    """Draw epigenomic track ribbons into a new axes and return it.

    Parameters
    ----------
    loci : sequence
        ``(chrom, start, end)``, 0-based half-open.
    feature : list of str, default []
        Keys of ``epi_key_dict``: ``'dnase'``, ``'atac'``, ``'k27ac'``,
        ``'k4me3'``, ``'k27me3'``, ``'k9me3'``, ``'ctcf'``.  Unknown keys raise
        ``KeyError``.
    figsize : tuple, default ``(5, 3)``
        Figure size in inches.

    Returns
    -------
    matplotlib.axes.Axes
        The axes, carrying one ``fill_between`` ribbon per requested feature.
        The figure is owned by the caller, which makes this the variant to use
        when composing multi-panel layouts.

    Notes
    -----
    * BigWig paths are hard-coded to ``/cluster2/huanglab/liquan/data/K562/``.
    * Each ribbon spans ``[y_bottom, y_bottom + signal / max(signal)]`` and
      ``y_bottom`` advances by 0.5 per feature; the y axis is therefore a stack
      index and the y ticks are hidden.
    * An all-zero track produces ``nan`` and an invisible ribbon (no guard
      against division by zero).
    * Unlike :func:`draw_loci_context` this does not call ``plt.show()``.
    """''')
D(G, 484, '''    # Mark name -> BigWig file stem; the directory is hard-coded to the lab
    # cluster.
''')
D(G, 499, '''        # Ribbon over [y_bottom, y_bottom + signal/max], then raise the baseline
        # by 0.5 for the next track.
''')

S = 'D:/GitHub/eScreen/escreen/serve.py'
D(S, 5, '''
Interface surface (this module has no HTTP routes)
-----------------------------------------------
``serve.py`` is a **Streamlit** application, not a REST service: it defines no
``@app.route``, no FastAPI/Flask handler and no JSON-over-HTTP endpoint.  What it
exposes is a browser UI with three tabs, served by Streamlit's own websocket
protocol on ``--port`` (default 8501):

``Predict``
    Widgets: file uploader (``bed``/``tsv``/``txt``; headerless TSV whose first
    two columns are ``sequence`` and ``celltype``), ``Batch size`` number input
    (8-512, default 128), ``Predict`` button.  On submit the uploaded table is
    echoed with an extra ``prediction`` column and offered as
    ``predictions.tsv``.  Backed by ``eScreen_vX.predict``.

``Attribution``
    Widgets: free-text ``DNA sequence`` (truncated to 512 bp), free-text
    ``Cell type`` resolved by case-insensitive prefix matching against the known
    cell lines (exact single match is used directly, several matches produce a
    ``Suggestions`` selectbox, no match shows an error), ``Run attribution``
    button.  Output: a line chart of per-nucleotide importance.  Backed by
    Captum ``IntegratedGradients``.

``Train``
    Widgets: file uploader (TSV with ``sequence``, ``celltype``, ``label`` and an
    optional 4th ``score`` column), an expander with ``Epochs``,
    ``Learning rate``, ``Batch size``, ``Weight decay``, ``Task``
    (``reg``/``cls``), ``Boost resampling``, ``Boost temperature``,
    ``Early stop patience``, ``Aux BCE lambda``, ``Pos weight``, and a
    ``Start training`` button.  On submit it calls ``eScreen_vX.fit`` and writes
    ``data/finetuned_model.pt``.

CLI
---
``python -m escreen.serve --hostname 0.0.0.0 --port 8501`` re-executes this file
under ``streamlit run`` with ``_ESCREEN_SERVE=1`` in the environment; the
module-level block guarded by that flag is the actual application, and importing
the module has no side effects.

Relation to the manuscript
--------------------------
Supplementary Methods section 8 describes the *deployed* eScreen web server: a
Python web framework exposing RESTful API services, Tabix-indexed BED files for
CRE predictions and transcripts, BigWig tracks fetched on demand, and gene /
cCRE-ID / coordinate query modes over a 530,527-CRE compendium.  None of that is
implemented in this file.  This module is the local GUI for running the model
itself (predict, explain, fine-tune) and performs no indexed genomic retrieval;
section 8 and this module are different artefacts.
''')

D(S, 18, '''# Repository root and bundled data directory.  Every artefact (MEME file,
# checkpoint, Geneformer weights) is resolved relative to this, so the app can be
# launched from any working directory.
''')
D(S, 21, '''# One-hot channel order used by every model in this package: A, C, G, T.
''')

D(S, 25, '''    """One-hot encode a DNA string into a fixed ``max_len`` x 4 uint8 matrix.

    Parameters
    ----------
    seq : str
        Nucleotide string.  Converted with ``.upper()`` and truncated to
        ``max_len``.
    max_len : int, default 512
        Number of positions to allocate - the eScreen input width
        (Supplementary Methods section 2.3.1: cCRE centre +/- 256 bp).

    Returns
    -------
    numpy.ndarray
        Shape ``(max_len, 4)``, dtype ``uint8``, channel order A, C, G, T.
        Short sequences are zero-padded at the tail, so only the first
        ``min(len(seq), max_len)`` rows are meaningful.

    Notes
    -----
    Characters outside A/C/G/T - ``N``, IUPAC codes, whitespace - leave an
    all-zero row rather than raising, matching the model's treatment of masked
    bases.  Silent truncation means a sequence longer than 512 bp is accepted
    without warning.
    """''')

D(S, 34, '''    """Build the eScreen model, its PWM front end and its cell-type embeddings.

    Cached with ``@st.cache_resource``, so the heavy objects are constructed once
    per process and shared across reruns and browser sessions.

    Returns
    -------
    model : escreen.model.eScreen_vX
        Model on ``DEVICE`` in ``eval()`` mode, warm-started from
        ``data/best_model.pt`` when that file exists.
    DEVICE : str
        ``'cuda'`` when available, otherwise ``'cpu'``.
    token2GFcelltype : dict
        Integer cell-type token -> cell-line name.
    GFcelltype2token : dict
        Cell-line name -> integer token (the inverse mapping).

    Notes
    -----
    * Motif kernels come from ``data/consensus_pwms.meme`` via
      ``escreen.motif.load_pwm_from_meme_c(..., max_length=35)``, i.e. the
      forward/reverse-complement PWM pairs padded to 35 bp described in
      Supplementary Methods section 2.3.6, assigned as the frozen Conv1d weights.
    * Cell-type conditioning uses the Geneformer embeddings of
      ``data/DepMap_CellLine_embedding.pkl`` (mean-pooled DepMap bulk RNA-seq,
      Supplementary Methods section 2.3.5) plus the Geneformer V1-10M model and
      its 30M gene dictionaries under ``data/geneformer/``.
    * ``num_filters=256`` is the width of the learnable projection applied to the
      motif activations; the number of motif channels itself is implied by the
      PWM tensor (1,274 filters), not by this argument.
    * ``load_state_dict(..., strict=False)`` tolerates missing and extra keys
      silently, so a mismatched checkpoint degrades to partly random weights with
      no warning.  Inspect the returned ``_IncompatibleKeys`` if that matters.
    * All ``freeze_*`` flags are False, so the loaded model stays fully trainable:
      right for the Train tab, slower than a frozen encoder for pure inference.
    * ``DEVICE`` is chosen here, not by the caller, and is returned so the rest of
      the app can move tensors consistently.
    """''')
D(S, 41, '''    # Motif kernels: forward and reverse-complement PWMs padded to 35 bp - the
    # fixed Conv1d weights of Supplementary Methods section 2.3.6.
''')
D(S, 46, '''    # Geneformer cell-type embeddings (Supplementary Methods section 2.3.5) and
    # the name <-> token maps used to condition every forward pass.
''')
D(S, 79, '''    # Optional warm start from the shipped checkpoint; strict=False tolerates
    # architecture drift instead of failing.
''')

D(S, 91, '''    """Convert a sequence/cell-type table into the dict the model consumes.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have ``sequence`` and ``celltype`` columns.
    ct2idx : dict
        Cell-line name -> integer token.  Mutated in place: unseen names are
        appended with index ``len(ct2idx)``.
    seq_len : int, default 512
        One-hot width per sequence.

    Returns
    -------
    data : dict
        ``{'sequence': ndarray (n, seq_len, 4) uint8, 'cell_type': ndarray (n,)
        int32, 'length': n}`` - exactly the layout ``eScreen_vX.predict`` and
        ``.fit`` index into.
    ct2idx : dict
        The (possibly extended) name -> token mapping.

    Notes
    -----
    Both call sites pass a fresh ``{}``, so tokens are assigned in
    first-appearance order instead of matching the Geneformer token ids: the
    cell-type conditioning is then arbitrary unless the uploaded names happen to
    line up.  Pass ``token2GFcelltype`` to reuse the pretrained mapping.
    """''')

D(S, 107, '''    """Integrated Gradients attributions, reduced to one score per nucleotide.

    Implements the attribution of Supplementary Methods section 3.2: all-zero
    baseline, Gauss-Legendre numerical integration with 30 steps, then the
    completeness step (attribution multiplied element-wise by the input).

    Parameters
    ----------
    model : eScreen_vX
        Model whose ``forward(x, ct)`` returns one scalar per sample.  Set to
        ``eval()`` here regardless of the caller's state.
    seqs : numpy.ndarray
        One-hot input, shape ``(n, 512, 4)``; cast to float32 on ``device``.
    ct_indices : numpy.ndarray
        Cell-type tokens, shape ``(n,)``, passed as
        ``additional_forward_args``.
    device : str
        Torch device.
    batch_size : int, default 8
        Sequences per IG call; the cost is ``n_steps`` forward/backward passes
        per batch.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 512)``, float32.  Each entry is the sum over the four
        nucleotide channels of ``attribution * input``; because the input is
        one-hot, only the channel of the present base contributes, so the value
        is that base's signed importance.

    Notes
    -----
    * The convergence delta returned by Captum is discarded: nothing checks that
      the 30-step Gauss-Legendre approximation converged.
    * No CRE-internal normalisation is applied here.  The paper's
      ``norm_mean`` (motif-interval mean divided by whole-CRE mean,
      Supplementary Methods section 3.3.2) belongs downstream.
    * ``eScreen_vX.forward`` accepts ``(x, ct, return_cls, return_emb)``, so IG
      explains the default regression output (``return_cls=False``).
    * Memory scales with ``batch_size`` because gradients are retained on the
      expanded interpolation batch.
    """''')
D(S, 115, '''        # IG from the all-zero baseline with Gauss-Legendre quadrature over 30
        # steps: Supplementary Methods section 3.2.
''')
D(S, 120, '''        # Completeness step: attribution x input, summed over the 4 channels.
''')

D(S, 126, '''    """CLI entry point: relaunch this file under ``streamlit run``.

    The function does not serve anything itself.  It re-invokes the current
    interpreter with ``-m streamlit run <this file>``, forwarding the host and
    port and forcing headless mode, and exports ``_ESCREEN_SERVE=1`` so that the
    module-level app block executes in the child process.  This double-launch is
    what lets ``python -m escreen.serve`` behave like a single command.

    Parameters
    ----------
    (none)
        Parsed from ``argv``: ``--hostname`` (default ``0.0.0.0``) and
        ``--port`` (default ``8501``).

    Returns
    -------
    None
        Blocks until the Streamlit child process exits.

    Notes
    -----
    ``subprocess.run`` is called without ``check=True``, so a failing Streamlit
    child leaves the parent with a zero exit status - scripted callers cannot
    detect a crashed server from the return code.
    """''')
D(S, 133, '''    # Re-exec under `streamlit run`; the environment flag below makes the
    # module-level block execute in the child process only.
''')

D(S, 141, '''# ---------------------------------------------------------------------------
# Streamlit app body.  Executes only in the child process started by main(),
# which is why importing escreen.serve has no side effects.  Structure: one
# cached model (init_model), then three tabs - Predict, Attribution, Train.
# Streamlit reruns this whole block on every widget interaction.
# ---------------------------------------------------------------------------
''')
D(S, 145, '''# Build the model once per process: the checkpoint load and Geneformer init are
# the expensive part, so they must not re-run per widget.
''')
D(S, 148, '''# Three tabs, one per entry point of the package: inference, attribution and
# fine-tuning.
''')

D(S, 150, '''    # ---- Predict: (sequence, celltype) rows -> activity score -------------
''')
D(S, 160, '''            # Headerless TSV: the first two columns are taken as
            # sequence/celltype, everything else in the file is ignored.
''')

D(S, 171, '''    # ---- Attribution: Integrated Gradients over one 512 bp sequence ------
''')
D(S, 186, '''                # Case-insensitive prefix match against the known cell lines;
                # ambiguous prefixes must be disambiguated before running.
                # 'attr_ct' stays None when nothing matches, which disables the
                # button's effect below.
''')

D(S, 211, '''    # ---- Train: fine-tune the loaded checkpoint on a labelled table ------
''')
D(S, 238, '''            # Layout: sequence, celltype, label[, score].  'score' (continuous)
            # becomes the regression target; 'label' (binary) drives the
            # auxiliary BCE head and the boosted sampler.
''')
D(S, 249, '''            # Regression target: the continuous score when present, otherwise
            # the binary label cast to float.
            # has_binary gates the auxiliary classification target: a label
            # column with more than two distinct values cannot supervise BCE and
            # is then used only through 'y'.
''')
D(S, 257, '''            # fit() owns the boosting sampler, the smooth-L1 + auxiliary BCE
            # composite loss, per-epoch checkpointing and early stopping.
''')
D(S, 272, '''            # Persist the fine-tuned weights next to the shipped checkpoint.
''')
