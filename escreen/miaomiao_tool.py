"""Numeric helpers and epigenomic-signal utilities for the eScreen-local pipeline.

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
"""
import torch
import numpy as np
import pandas as pd
from collections import Counter

def match_summit(data,summit_ref,threshold=200):
    """Assign each locus to its nearest epigenomic summit within a distance cut-off.

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
    * Progress is printed with ``\r`` on a single line.
    """
    for i in data.index.to_list():
        chromo=data.loc[i,'chr']
        ref_sub=summit_ref[summit_ref['chr']==chromo]
        # Unsigned distance from the element centre to each summit on the same
        # chromosome; the nearest one wins regardless of direction.
        ref_sub['delta']=abs(ref_sub['start']-data.loc[i,'loci'])
        ref_sub=ref_sub.sort_values(by='delta')
        
        if len(ref_sub)>0:
            if ref_sub.loc[ref_sub.index.to_list()[0],'delta']>threshold:
            # Beyond the cut-off: mark the element as 'loss' and use large
            # sentinels so these rows sort last rather than acting as if they had
            # a real zero-distance summit hit.
                data.loc[i,'summit_group']='loss'
                data.loc[i,'summit_value']= 1e5
                data.loc[i,'summit_delta']=ref_sub.loc[ref_sub.index.to_list()[0],'delta']
            else:
                data.loc[i,'summit_group']=ref_sub.loc[ref_sub.index.to_list()[0],'peak_name']
                data.loc[i,'summit_value']=ref_sub.loc[ref_sub.index.to_list()[0],'value']
                data.loc[i,'summit_delta']=ref_sub.loc[ref_sub.index.to_list()[0],'delta']
        else:
            data.loc[i,'summit_group']='loss'
            data.loc[i,'summit_value']= 1e5
            data.loc[i,'summit_delta']= 1e8
            
        print(f'{i}/{len(data)}',end='\r')
    return data

def capture_summit(data, summit_ref, bw_dict, 
                   epi='', sequence_len=200, kmer=40):
    """Store a summit-normalised local signal profile for every bin.

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
    """
    nearest_summits = []
    for chromo in data['chr'].unique():
        summit_subset =summit_ref[chromo]
        if not summit_subset.empty:
            for _, b_row in data[data['chr'] == chromo].iterrows():
                # Distance to the bin CENTRE, not the bin start; both bins and
                # summits are 0-based half-open coordinates.
                distances = np.abs(summit_subset['start'] - (b_row['start']+b_row['end'])/2 )
                nearest_index,nearest_distance = distances.idxmin(),distances.min()
                # Only summits within 500 bp of the bin centre are recorded;
                # all other bins keep the 1e6 fill applied below.
                if nearest_distance<500:
                    nearest_summits.append((b_row.name, summit_subset.loc[nearest_index]))

    for b_index, nearest in nearest_summits:
        data.loc[b_index, f'{epi}_nearest_summit'] = nearest['value']

    # Unmatched bins get 1e6, so dividing by this sentinel drives the weight to
    # ~0 instead of raising a division by zero.
    data=data.fillna(1e6)

    summit_array=[]
    for i in data.index.to_list():
        # One ratio per offset k in [0, sequence_len - kmer]:
        # signal(start + k) / summit height, i.e. a summit-normalised local
        # profile.  np.newaxis adds the channel axis, squeezed away immediately.
        _a_=np.array([ catch_epi( bw_dict[epi],(data.loc[i,'chr'],data.loc[i,'start']+k) )/data.loc[i,f'{epi}_nearest_summit'] \
                       for k in range(int( sequence_len-kmer+1) ) ])[np.newaxis,:,:] 
        summit_array.append(_a_.squeeze())
    data[f'{epi}_summit_weight']=summit_array

    return data

def get_value(data,bw,feature='profile',threshold=0.5,_filter_=True,_verbose_=True):
    
    """Attach raw BigWig profiles to an interval table with a per-row sanity flag.

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
    """
    if data is None:
        pass
    else:
        array_house=[]
        ok_array=[]
        for i in data.index.to_list():
            c=data.loc[i,'chr']
            a,b=min(data.loc[i,'start'],data.loc[i,'end']),max(data.loc[i,'start'],data.loc[i,'end'])
            try:
                array=np.array( bw.values(c,a,b) )
                array[np.isnan(array)] = 0
                array_house.append( array )
                # Degeneracy check: how much of the interval is occupied by a
                # single repeated value (an all-zero stretch, a masked region, ...).
                element_counts = Counter(array)
                most_common_element, most_common_count = element_counts.most_common(1)[0]
                # NOTE (documented, not fixed): the second disjunct makes this
                # True for every non-empty selection, so 'ok' effectively only
                # marks rows whose BigWig query failed.
                ok_array.append( (most_common_count <= (b - a) *threshold) | (most_common_count !=0 ) )
            except:
                # Failed query: substitute a 200-bp zero vector regardless of
                # the real interval length, and flag the row as not ok.
                array_house.append( np.zeros(200) )
                ok_array.append(1<0)
        data[feature]=array_house
        data['ok']=ok_array
        if _filter_:
            data=data[data['ok']]
        if _verbose_:
            print(len(data[data.y>0]),end="/")
            print(len(data[data.y==0]))
    
    return data

def ln(x):
    """Natural logarithm: ``log(x) / log(e)``."""
    return np.log(x)/np.log(np.e)

def lg(x):
    """Base-10 logarithm via the change-of-base formula: ``log(x) / log(10)``."""
    return np.log(x)/np.log(10)

def sigmoid_normalization_point(array,s1=(0.0,0.3),s2=(-0.3,0.7)):

    """Fit a sigmoid through two anchor points and apply it to an array.

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
    """
    e=2.71828
    
    # Solve y = 1 / (1 + exp(-n*x + b)) through the two anchors, i.e. the logit
    # of the anchor outputs is linear in x:
    #     logit(y1) = -n*x1 + b        logit(y2) = -n*x2 + b
    # hence n = (logit(y1) - logit(y2)) / (x2 - x1) and b from either anchor.
    x1,y1=s1
    x2,y2=s2
    
    n= ( ln(1/y1-1)-ln(1/y2-1) )/( x2-x1 )
    b= ln(1/y1-1)+n*x1 
    
    print(f'parameter: n={n:.4f},b={b:.4f}')
    
    # Apply with the same sign convention used when solving: exp(-n*x + b).
    n_array=1/( 1+e**(n*-1*array + b ) )
    
    return n_array

def nonzero_mean(arr, axis):

    """Mean over an axis that ignores zeros (zeros are missing, not measurements).

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
    """
    arr_with_nan = np.where(arr != 0, arr, np.nan)

    mean_result = np.nanmean(arr_with_nan, axis=axis)
    
    return mean_result

def norml(x):
    """Min-max normalise to [0, 1].

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
    """
    if isinstance(x, torch.Tensor):
        x = x.to('cpu')
        x = x.numpy()
    else:
        x=np.array(x)
        
    return (x-x.min())/(x.max()-x.min())

def safe_divide(a, b):
    """Element-wise division in which a zero denominator yields 0.

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
    """
    a = np.array(a)
    b = np.array(b)
    result = np.zeros_like(a, dtype=np.float64)
    non_zero_mask = b != 0
    result[non_zero_mask] = a[non_zero_mask] / b[non_zero_mask]
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        return pd.Series(result)
    return result

def safe_log(a, base=2):
    """Element-wise logarithm with 0 for non-positive inputs.

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
    """
    a = np.array(a)  
    result = np.zeros_like(a, dtype=np.float64)
    positive_mask = a > 0
    result[positive_mask] = np.log(a[positive_mask]) / np.log(base)  # use the change-of-base formula
    if isinstance(a, pd.Series):
        return pd.Series(result)
    return result

def remove_overlap(df1, df2):
    """Anti-join two frames on exact row identity.

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
    """
    overlap = pd.merge(df1, df2, how='inner')
    df1_non_overlap = df1[~df1.apply(tuple, 1).isin(overlap.apply(tuple, 1))]
    df2_non_overlap = df2[~df2.apply(tuple, 1).isin(overlap.apply(tuple, 1))]
    return df1_non_overlap, df2_non_overlap


def power_norm(data, max_val=10, gamma=0.424):
    """Gamma-compress absolute values onto [0, 1].

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
    """
    abs_data = np.abs(data)
    scaled = (abs_data / max_val) ** gamma
    # Clip bounds the scale; |data| discards sign.
    return np.clip(scaled, 0, 1)


