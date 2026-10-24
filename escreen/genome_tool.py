"""Genomic interval, sequence and signal-track helpers shared by the eScreen analyses.

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
  ``/path/to/data/K562/DNase.bigWig``;
* :func:`merge_intervals` references ``BedTool``,
  :func:`genome_sequence_withoutcoding` references ``rainbow_tqdm``, and
  :func:`eqtlboxplot_draw`` references ``time`` - none of these three names is
  imported in this module, so those functions raise ``NameError`` unless the
  caller injects the symbols into the module namespace.

These defects are documented rather than repaired, because the module's runtime
behaviour (and every figure already produced from it) must stay unchanged.
"""
from pyliftover import LiftOver
import pyBigWig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

# Nucleotide -> one-hot channel.  ``N`` maps to all zeros (a fully "off"
# position) rather than to a fifth channel - the same convention the model uses
# for masked/padding bases and for the all-zero IG baseline.  Any other character
# (lower case, IUPAC ambiguity codes, gaps) makes one_hot() raise KeyError.
base_to_array={'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
# Canonical chromosomes only: chrY, chrM and alt/random/decoy contigs are absent,
# so every helper that filters on this list silently drops rows on those contigs.
chr_list=[f'chr{i}' for i in range(1,23)]+['chrX']
# Complement table used by reverse_seq(); N is self-complementary.
trans_dict={'A':'T','C':'G','G':'C','T':'A','N':'N'}

def reverse_seq(seq):
    """Complement a nucleotide string in place order (not a reverse complement).

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
    """
    return ''.join([trans_dict[b] for b in seq])

def one_hot(seq):
    """One-hot encode a nucleotide string.

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
    """
    return np.stack( np.array( [base_to_array[a] for a in seq] ) )

def catch_epi(bw=None,loci=[]):
    """Read a BigWig track over a locus, zero-padding anything off the chromosome.

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
    """
    if bw==None:
        print('No match tasks.')
        return np.zeros(1)
    
    if len(loci)<1:
        print('No data input.')
        return np.zeros(1)
    
    try:
    # Direct query: valid when the interval lies fully inside the chromosome.
        values = bw.values( loci[0],int(loci[1]),int(loci[2]) )
    except:
        # Rebuild the out-of-range part from zeros so that the returned length
        # is always end - start; 'bound' is the chromosome length.
        bound = int(bw[loci[0]])
        if int(loci[1])<0 and int(loci[2])<bound:
            values = np.concatenate([
                np.zeros(abs(int(loci[1]))),np.array( bw.values( loci[0],0,int(loci[2]) ) )
            ])
        elif int(loci[1])>=0 and int(loci[2])>bound:
            values = np.concatenate([
                np.array( bw.values( loci[0],int(loci[1]),bound ) ),np.zeros(int(loci[2])-bound)
            ])
        elif int(loci[1])<0 and int(loci[2])>bound:
            values = np.concatenate([
                np.zeros(abs(int(loci[1]))),np.array( bw.values( loci[0],0,bound ) ),np.zeros(int(loci[2])-bound)
            ])            
    return values

def genome_sequence(data,fa,chr_list=chr_list):
    """Extract hg38 sequence for every interval and one-hot encode it.

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
    """
    data=data[data['chr'].isin(chr_list)]
    # Drop non-canonical contigs first so every kept row can be sliced from the
    # FASTA; the returned array follows data.index and may be shorter than
    # len(data).
    seq_input=np.stack([ 
        one_hot( 
        # pyfaidx slicing is 0-based half-open (BED convention), and .upper()
        # converts soft-masked bases to hard ACGT.
            fa[ data.loc[i,'chr']][ data.loc[i,'start']:data.loc[i,'end'] ].seq.upper()
        ) for i in tqdm(data.index.to_list())
    ])
    return seq_input

def _genome_sequence_withoutcoding_(data,fa,chr_list=chr_list):
    """Extract raw interval sequences, without one-hot encoding.

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
    """
    data=data[data['chr'].isin(chr_list)]
    seq_input=[ fa[ data.loc[i,'chr'] ][ data.loc[i,'start']:data.loc[i,'end'] ].seq for i in data.index.to_list()]
    return seq_input

def genome_sequence_withoutcoding(data,fa,chr_list=chr_list):
    """Same as :func:`_genome_sequence_withoutcoding_`, with a progress bar.

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
    """
    data=data[data['chr'].isin(chr_list)]
    seq_input=[ fa[ data.loc[i,'chr'] ][ data.loc[i,'start']:data.loc[i,'end'] ].seq for i in rainbow_tqdm(data.index.to_list())]
    return seq_input

def loci_split(df,length=200):
    """Tile intervals into fixed-width bins of ``length`` bp.

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
    """
    result = []
    for _, row in df.iterrows():
        chrom = row['chr']
        start = row['start']
        end = row['end']

        while start < end:
            # Full-width bin: emit [start, start + length) and advance.
            if end-start>length:
                segment_end = min(start + length, end)
                result.append({'chr': chrom, 'start': start, 'end': segment_end})
                start = segment_end
            else:
            # Final, short bin: padded up to a full 'length' bp (it may overshoot
            # the original end) so every bin has the same width.
                segment_end = max(start + length, end)
                result.append({'chr': chrom, 'start': start, 'end': segment_end})
                start = segment_end
    split_df = pd.DataFrame(result)
    print(len(split_df))
    return split_df

def merge_intervals(df):
    """Sort and zero-gap merge BED intervals with bedtools, after light repair.

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
    """
    df=df.reset_index(drop=True)
    
    for i in range(len(df)):
          
        # Repair inverted intervals produced by upstream parsing.
        if df.loc[i,'start']>df.loc[i,'end']:
            s,e=df.loc[i,'start'],df.loc[i,'end']
            df.loc[i,'start'],df.loc[i,'end']=e,s

        # bedtools merge drops zero-length intervals, so anything under 3 bp is
        # widened to 20 bp first (20 is literal here, unrelated to the 3 bp test).
        if df.loc[i,'end']-df.loc[i,'start']<3:
            df.loc[i,'end']=df.loc[i,'start']+20
    
    # sort | merge with bedtools defaults: overlapping AND bookended intervals
    # are collapsed (zero-gap merge, Supplementary Methods section 2.4.3).
    # Temporary files land in the current working directory.
    df[['chr','start','end']].to_csv('intervals.bed',sep='\t',index=False,header=None)
    a = BedTool('intervals.bed').sort()
    merged = a.merge()
    merged.saveas('merged_intervals.bed')
    merged_df=pd.read_csv('merged_intervals.bed',sep='\t',header=None,names=['chr','start','end'])
    
    os.remove('intervals.bed')
    os.remove('merged_intervals.bed')
    
    return merged_df

def convert_genome(region,lo=None):
    """Lift one interval between genome builds, nudging the start when needed.

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
    """
    if not lo is None:
        a,b,c=region[1],region[2],region[0]
        # d is the interval length preserved across builds; only the start is
        # looked up, so a straddling interval keeps its length, not its end.
        d=b-a
        status=0
        count = 0
        while (status<1):
            converted = lo.convert_coordinate(c, a, b)        
            if not converted is None:
                if len(converted)>0:
                    chrom,s = converted[0][0],converted[0][1]
                    status=1
                else:
                    # Unmappable here: shift the whole interval one base
                    # downstream and retry (capped at 100 attempts below).
                    a+=1;b+=1;count+=1
                    if count > 100: 
                        print('Loci can not be converted:',end=' ');print(region)
                        return '0',0,0
                    
            else:
                a+=1;b+=1;count+=1
                if count > 100: 
                    print('Loci can not be converted:',end=' ');print(region)
                    return '0',0,0
                
            if status==1:
                return chrom,s,s+d
            
            
def ref_filter(r, d):
    """Keep the reference intervals that overlap at least one query interval.

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
    """
    df_list = []
    outref_list = []
    d_grouped = {chr_: df for chr_, df in d.groupby('chr')}

    for chromo, s, e in zip(r['chr'], r['start'], r['end']):
        if chromo in d_grouped:
            d_chr = d_grouped[chromo]
            mask = (
            # Three overlap cases: query start inside, query end inside, or the
            # reference interval fully contained in the query.
                ((d_chr['start'].astype(int) >= s) & (d_chr['start'].astype(int) <= e)) |
                ((d_chr['end'].astype(int) >= s) & (d_chr['end'].astype(int) <= e)) |
                ((d_chr['start'].astype(int) <= s) & (d_chr['end'].astype(int) >= e))
            )
            ds = d_chr[mask]
            if not ds.empty:
                df_list.append(ds)
                outref_list.append(r.loc[r['chr'] == chromo])

    df = pd.concat(df_list) if df_list else pd.DataFrame()
    outref = pd.concat(outref_list) if outref_list else pd.DataFrame()

    if df.empty:
        print('No activity guide!')

    return outref,df 

def split_coordinates(df,col='Coordinates'):
    """Split a ``Coordinates`` string into ``chr``/``start``/``end``/``strand``.

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
    """
    df[col] = df[col].str.split('|').str[-1]
    split_df = df[col].str.split(':', expand=True)
    # '<tag>|chr:start-end:strand' -> take the last pipe-separated field.
    df['chr'] = split_df[0]
    region_split = split_df[1].str.split('-', expand=True)
    df['start'] = region_split[0].fillna(0).astype(int)
    df['end'] = region_split[1].fillna(0).astype(int)
    try:
        df['strand'] = split_df[2].fillna('*')
    except:
        print('All data loss strand info. Use * fill')
        df['strand'] = '*'
    
    return df[['chr', 'start', 'end', 'strand']]


def find_contextual(data,fa=None,expand_flank=30,upstream=4,downstream=6,total_len=30):

    """Locate a 20 bp sequence inside a flanked window and return its 30 bp context.

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
    """
    direction=[];seq_adj=[];position_adj=[]
    
    merge_table=data
    
    for _,row in tqdm(merge_table.iterrows(),total=len(merge_table)):
    
        # Only the last 20 bp of the stored sequence serve as the search key;
        # spaces (alignment padding) are stripped first.
        seq = row['sequence'].upper().replace(' ','')[-20:]

        try:
            c,s,e = row['chr'],int(row['start']),int(row['end'])
            s,e = min(s,e),max(s,e)
            if not c in chr_list:
                direction.append('-1')
                seq_adj.append('N'*total_len)
                position_adj.append(-1)
                continue
        except:
            direction.append('-1')
            seq_adj.append('N'*upstream+s1+'N'*downstream)
            position_adj.append(-1)
            continue

        # Search window: +/- expand_flank around the annotated interval,
        # 0-based half-open because pyfaidx is.
        sequence=fa[c][s-expand_flank:e+expand_flank].seq.upper()

        # Forward-strand match first, then the reverse complement.
        l1=sequence.find(seq)

        if l1<0:
            l1=sequence.find(reverse_seq(seq)[::-1])
            if l1<0:
                # Neither strand matched: fall back to a fixed 30 bp window
                # starting 'upstream' bp before the annotated start.
                direction.append('+');seq_adj.append( fa[c][s-upstream:s+20+downstream].seq.upper() )
                position_adj.append(s)
            else:
                direction.append('-');seq_adj.append( reverse_seq( fa[c][s-expand_flank+l1-downstream:s-expand_flank+l1+20+upstream].seq.upper() )[::-1] )
                position_adj.append(s-expand_flank+l1)
        else:
            direction.append('+');seq_adj.append( fa[c][s-expand_flank+l1-upstream:s-expand_flank+l1+20+downstream].seq.upper() )
            position_adj.append(s-expand_flank+l1)
    
    return direction,seq_adj,position_adj


    
def eqtlboxplot_draw(data,chromo,lo=None,label='eQTL boxplot figure',label_list=[],guide_expand=False,expand=90,sample_limit=200):

    """Count eQTLs per group interval and draw the group-wise comparison panel.

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
      ``/path/to/data/eQTL/ciseqtl-loci.txt`` with columns
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
    """
    min_start=min( [data[i].start.min() for i in range(len(data)) ] )
    max_end=max( [data[i].end.max() for i in range(len(data)) ] )
    
    # Hard-coded eQTL table (not portable).  SNPChr is a bare integer (1..22) and
    # SNPPos is 1-based, which is why the counting test below is inclusive on both
    # ends.
    eqtlx=pd.read_csv("/path/to/data/eQTL/ciseqtl-loci.txt",sep='\t')
    
    import concurrent.futures

    def process_group(ii,guide_expand=guide_expand,expand=expand):
        """Count eQTL variants for one group of intervals (thread-worker body).

        Runs inside a ``ThreadPoolExecutor`` with one call per group, so `ii` is
        a whole DataFrame rather than a row.

        Parameters
        ----------
        ii : pandas.DataFrame
            One group's intervals, with ``chr``, ``start``, ``end``.  Mutated in
            place with the ``snp`` count column.
        guide_expand : bool
            Bound from the enclosing function; widens each interval by
            ``expand`` bp on both sides when True.
        expand : int
            Bound from the enclosing function; expansion in bp.

        Returns
        -------
        pandas.DataFrame
            ``ii`` with ``snp`` added: per-interval eQTL variant counts, ``NaN``
            for chrX, 0 for degenerate (zero-length) or unparsable intervals.

        Notes
        -----
        * ``itertuples(index=False, name=None)`` yields plain
          ``(chrom, start, end)`` tuples, which keeps the loop cheap but loses
          the row index; the counts are therefore assigned positionally, in the
          same order as ``ii``.
        * ``results = {}`` and ``i = 0`` are dead locals left over from an earlier
          revision; only ``groupresults`` is used.
        * Progress is printed from every thread with ``\r``, so the line counter
          interleaves when several groups run concurrently.
        """
        groupresults = []
        intervals = list(ii[['chr','start','end']].itertuples(index=False, name=None))
        results = {}
        i = 0

        for chrom, start, end in intervals:
            
            # chrX is skipped because SNPChr is matched via int(chrom[3:]).
            if chrom=='chrX':
                groupresults.append(np.nan)
                continue
            
            if guide_expand:
                start=start-expand
                end=end+expand
            
            if not lo is None:
                converted = lo.convert_coordinate(chrom, start, end)
                chrom = converted[0][0]
                s, e = converted[0][1], converted[0][3]
                # Reversed conversion (s > e) means the interval straddled an
                # assembly gap; shift the original interval instead of trusting
                # the converted pair.
                if abs(s - e) > abs(start - end) * 1.1:
                    if s > e:
                        delta = e - end
                        start = start + delta
                        end = e
                    else:
                        delta = s - start
                        end = end + delta
                        start = s
                else:
                    start, end = s, e


            if start == end:
                groupresults.append(0)
            else:
                start, end = min(start, end), max(start, end)
                print(start, end)
                try:
                    # Inclusive on both ends; see the coordinate-convention note
                    # at the top of this function.
                    groupresults.append(eqtlx[(eqtlx.SNPChr==int(chrom.split('chr')[1])) & (eqtlx.SNPPos >= start) & (eqtlx.SNPPos <= end)].shape[0])
                except:
                    print(f'bad region: {chrom}-{start}-{end}')
                    groupresults.append(0)

            i += 1
            print(f'{i}/{len(ii)}\r', end="")


        ii['snp'] = groupresults
        print('')
        return ii

    groups = data

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_group, groups))

    for i,(j,k) in enumerate( zip(data,results) ):
        j=k
        j['label']=label_list[i]

    combined_df = pd.concat(data).reset_index(drop=True)

    t1, p1 = ttest_ind(data[0]['snp'], data[1]['snp'])

    plt.figure(figsize=(4, 3.2))
    sns.set_style("ticks")

    sns.boxplot(x='label', y='snp', data=combined_df,
                palette=['#ecf0f1'], showfliers=False, width=0.4)
    sns.stripplot(x='label', y='snp', data=combined_df,
                  jitter=True, alpha=0.4, color='#74b9ff', size=2)

    sf=1.2
    figuretop=max(data[0].snp.max(),data[1].snp.max())+2
    plt.ylim(-1,figuretop)

    x1, x2 = 0, 1 
    y, h, col = figuretop - 1.0*sf, 0.1, 'k'
    if data[0]['snp'].mean()>data[1]['snp'].mean():
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='#eb2f06')
    else:
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='#0984e3')
    plt.text((x1 + x2) * .5, y + h, f'p = {p1:.3g}', ha='center', va='bottom', color=col,fontsize=7)

    plt.title('eQTL enrichment cross groups')
    plt.xticks(fontsize=6,rotation=30)
    plt.ylabel('eQTL loci count')
    
    if label:
        plt.xlabel(label)
    else:
        plt.xlabel('Group')
        label=time.time()

    ax = plt.gca()

    ax.tick_params(axis='y', which='both', direction='in', length=4)    
    ax.tick_params(axis='x', which='both', direction='out', length=4)   
    
    plt.tight_layout()    
    plt.show()
    return None
    
    
def gwashistplot_draw(data,chromo,gwas_data,lo=None,label='GWAS figure',label_list=[],guide_expand=False,expand=90):
                                                 
    """Count GWAS variants per group interval and draw the group totals.

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
    """
    import concurrent.futures
    
    def process_group(ii,guide_expand=guide_expand,expand=expand):
        """Count GWAS variants for one group of intervals (thread-worker body).

        Runs inside a ``ThreadPoolExecutor`` with one call per group.

        Parameters
        ----------
        ii : pandas.DataFrame
            One group's intervals, with ``chr``, ``start``, ``end``.  Mutated in
            place with the ``gwas`` count column.
        guide_expand : bool
            Bound from the enclosing function; widens each interval by
            ``expand`` bp on both sides when True.
        expand : int
            Bound from the enclosing function; expansion in bp.

        Returns
        -------
        pandas.DataFrame
            ``ii`` with ``gwas`` added: per-interval variant counts, ``NaN`` for
            chrX, 0 for zero-length or unparsable intervals.

        Notes
        -----
        * The Liftover retry loop has no attempt cap: only a ``None`` return from
          ``convert_coordinate`` ends it (via ``status = 2``), so a locus that
          keeps yielding an empty list spins forever.
        * ``gwas_data.query(...)`` is evaluated per interval with local variables
          interpolated through ``@``, i.e. one full-table filter per interval -
          O(intervals x variants).  Pre-filtering by chromosome would be much
          cheaper, but the behaviour is left as-is.
        * ``results = {}`` and ``i = 0`` are dead locals from an earlier revision.
        """
        groupresults = []
        intervals = list(ii[['chr','start','end']].itertuples(index=False, name=None))
        results = {}
        i = 0
        for chrom, start, end in intervals:
            if chrom=='chrX':
                groupresults.append(np.nan)
                continue      
            if guide_expand:
                start=start-expand
                end=end+expand      
            status=0
            if not lo is None:
                # Keep nudging the start until the locus maps.  There is no
                # attempt cap here (unlike convert_genome): only a None return from
                # LiftOver breaks the loop, so an endlessly empty result hangs.
                while (status<1):
                    converted = lo.convert_coordinate(chrom, start, end)        
                    if not converted is None:
                        if len(converted)>0:
                            chrom = converted[0][0]
                            s, e = converted[0][1], converted[0][3]
                            status=1
                        else:
                            start+=1
                            end+=1
                    else:
                        groupresults.append(0)
                        status=2
                        break
                if status==2:
                    continue                  
                if abs(s - e) > abs(start - end) * 1.1:
                    if s > e:
                        delta = e - end
                        start = start + delta
                        end = e
                    else:
                        delta = s - start
                        end = end + delta
                        start = s
                else:
                    start, end = s, e
            if start == end:
                groupresults.append(0)
            else:
                start, end = min(start, end), max(start, end)
                try:
                    chrom_num = int(chrom.split('chr')[1])
                    # GWAS summary-table columns: chromosome (integer) and
                    # base_pair_location (1-based).
                    sub_gwas = gwas_data.query("chromosome == @chrom_num and base_pair_location >= @start and base_pair_location <= @end")
                    groupresults.append(sub_gwas.shape[0])
                except:
                    groupresults.append(0)
            i += 1
            print(f'{i}/{len(ii)}\r', end="")
        ii['gwas'] = groupresults
        print('')
        return ii

    groups = data
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_group, groups))
    
    for i,(j,k) in enumerate( zip(data,results) ):
        j=k
        j['label']=label_list[i]
    
    combined_df = pd.concat([data]).reset_index(drop=True)
    
    plt.figure(figsize=(4, 3.2))
    
    groups = [d['label'].unique()[0] for d in data ] 
    # One bar per group: total number of GWAS variants inside the group's
    # intervals (sum, not mean).
    counts = [d['gwas'].sum() for d in data ]
    positions = range(len(groups))
    
    plt.bar(positions, counts, color='skyblue', edgecolor='black', width=0.5)
    plt.xticks(positions, groups,fontsize=6)
    
    plt.title('Sum of GWAS loci cross group')
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    print(counts)
    
    
def draw_loci_context(loci):

    """Plot six K562 epigenomic tracks as offset, max-normalised profiles.

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
    * Tracks are hard-coded to ``/path/to/data/K562/*.bigWig``:
      DNase, ATAC, H3K27ac, H3K4me3, CTCF, H3K27me3 - the marks used for the
      peak nodes of the eScreen-local graph
      (Supplementary Methods section 2.4.2).
    * Each track is divided by its own maximum and shifted by a fixed offset
      (2.0, 1.5, 1.0, 0.5, 0.0, -0.5), so the y axis is a stack index, not a
      signal unit.  A flat zero track gives ``0 / 0 = nan`` and is drawn as a
      gap.
    * The x axis is one point per base over ``[start, end)``, so a multi-megabase
      window allocates one float per base per track.
    """
    chrchr=loci[0];startstart=loci[1];endend=loci[2]

    # Hard-coded K562 tracks from the lab cluster: the six marks behind the peak
    # nodes and the genome-context panels.
    bw=pyBigWig.open('/path/to/data/K562/DNase.bigWig')
    dnase_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/path/to/data/K562/ATAC.bigWig')
    atac_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/path/to/data/K562/H3K27ac.bigWig')
    k27ac_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/path/to/data/K562/H3K4me3.bigWig')
    k4me3_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()
    
    bw=pyBigWig.open('/path/to/data/K562/CTCF.bigWig')
    ctcf_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()

    bw=pyBigWig.open('/path/to/data/K562/H3K27me3.bigWig')
    k27me3_array=np.array(bw.values(chrchr,startstart,endend))
    bw.close()
    
    fig=plt.figure(figsize=(5,3))
    
    positions = list(range(startstart,endend))

    # Max-normalise each track and add a fixed offset so six profiles share one
    # axis; the y value is a stack index, not a signal value.
    plt.plot(positions, dnase_array/dnase_array.max()+2.0, color='#AFBA0F',alpha=0.35)
    plt.plot(positions, atac_array/atac_array.max()+1.5, color='#a29bfe',alpha=0.35)
    plt.plot(positions, k27ac_array/k27ac_array.max()+1.0, color='#70a1ff',alpha=0.35)
    plt.plot(positions, k4me3_array/k4me3_array.max()+0.5, color='#7bed9f',alpha=0.35)
    plt.plot(positions, ctcf_array/ctcf_array.max()+0.0, color='#778ca3',alpha=0.35)
    plt.plot(positions, k27me3_array/k27me3_array.max()-0.5, color='#778ca3',alpha=0.35)
    
    plt.xlim([startstart,endend])
    plt.xticks([startstart,int( (startstart+endend)/2 ),endend])
    plt.ylim([-1.0,3.5])
    
    plt.title('Genome Context')

    plt.xlabel('Genome loci')

    plt.show()


def draw_loci_context_ax(loci,feature=[],figsize=(5,3)):
    """Draw epigenomic track ribbons into a new axes and return it.

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
    * BigWig paths are hard-coded to ``/path/to/data/K562/``.
    * Each ribbon spans ``[y_bottom, y_bottom + signal / max(signal)]`` and
      ``y_bottom`` advances by 0.5 per feature; the y axis is therefore a stack
      index and the y ticks are hidden.
    * An all-zero track produces ``nan`` and an invisible ribbon (no guard
      against division by zero).
    * Unlike :func:`draw_loci_context` this does not call ``plt.show()``.
    """
    chrchr = loci[0]
    startstart = loci[1]
    endend = loci[2]

    # Mark name -> BigWig file stem; the directory is hard-coded to the lab
    # cluster.
    epi_key_dict={'dnase':'DNase','atac':'ATAC',
                  'k27ac':'H3K27ac','k4me3':'H3K4me3',
                  'k27me3':'H3K27me3','k9me3':'H3K9me3',
                  'ctcf':'CTCF'}
    y_bottom=0

    fig, ax = plt.subplots(figsize=figsize)
    positions = list(range(startstart, endend))

    for e in feature:
        
        bw = pyBigWig.open(f'/path/to/data/K562/{ epi_key_dict[e] }.bigWig')
        array = np.array(bw.values(chrchr, startstart, endend))
        bw.close()
        
        # Ribbon over [y_bottom, y_bottom + signal/max], then raise the baseline
        # by 0.5 for the next track.
        ax.fill_between(positions, y_bottom, array/array.max()+y_bottom, color='#95afc0',alpha=0.35)
        y_bottom+=0.5

    ax.set_xlim([startstart, endend])
    ax.set_xticks([startstart, int((startstart + endend) / 2), endend])
    
    ax.set_ylim([-0.05, y_bottom+1])
    ax.set_yticks([])

    ax.set_title('Genome Context')
    ax.set_xlabel('Genome loci')

    return ax
