#!/usr/bin/env python3
"""
Export P0-filtered 207,352 samples as a CSV preserving the original CREs format.

What this script does
---------------------
This is the dataset-construction step that both Enformer strategies and every
other P0 benchmark start from. It takes the eScreen training bundle (an 8-tuple
pickle), applies the benchmark's sample filter, and re-splits the result 7:2:1 at
the CRE-identifier level.

It implements two rules from the Supplementary Methods:

* **Negative downsampling** (§2.3.4 and §6.1): keep every positive CRE, and keep
  a negative only if its ``RRA < 0.5`` *and* it is a negative in at least two of
  the P0 cell lines ("negative samples shared by at least two cell lines among
  K562, HepG2, HT29, A375 and hPSC").
* **Locus-aware / identifier-level splitting** (§2.3.3 and §6.1): split on the
  unique CRE id, never on individual rows, so the same CRE observed in several
  cell lines cannot land in both train and test. Additionally the chr11
  LCR/validation locus (chr11:5,200,000-5,300,000) is pulled out into its own
  ``locus`` split and excluded from train/valid/test.

Where it differs from the manuscript
------------------------------------
* §2.3.3 quotes a 7:2:1 split of the *whole* eScreen dataset (145,067 / 41,531 /
  20,715 CREs); this script re-splits the P0-filtered subset only, using
  ``SPLIT_SEED = 42``.
* The manuscript says "excluded all sites involved in subsequent validation
  experiments and the LCR locus (chr11:5,250,000-5,300,000)". The code excludes a
  slightly wider window, chr11:5,200,000-5,300,000, and it does not exclude the
  other experimental loci (only the chr11 one is handled here).
* The "P0 filter" itself (five cell lines, WTC11 relabelled to hPSC) is not
  described as such in the manuscript; it is the benchmark's implementation
  detail, named after the paper's "P0" data freeze.

Expected on disk
----------------
Input (hard-coded absolute path, Clustered filesystem; adjust before rerunning
anywhere else)::

    /cluster/huanglab/sluo/sluoo/escreen/eScreen_training/datasets/
        TestData_20260509.randomCRE.4.pkl

    An 8-tuple ``(trainset, validset, testset, train_idx, valid_idx, test_idx,
    cre_sub, final_df)``. Only ``*_idx`` (positions into the CRE table) and
    ``cre_sub`` (the CRE table itself) are used; the tensors are ignored.

Output (relative to the working directory)::

    ./enformer_embeddings/p0_filtered_samples.csv
    ./enformer_embeddings/p0_filtered_samples.pkl

    Same columns as ``cre_sub`` plus ``split`` and ``original_index``.

This script is a one-shot data-export job: it runs top-to-bottom at import time
(no functions guarding the work), writes both files, and prints the split counts.
It must be run before ``extraEnformerEmb.py``.
"""
import pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths & Constants ──
# Hard-coded cluster path; see the module docstring. The pickle predates
# numpy 2.0, hence _NumpyCompatUnpickler below.
ORIGINAL_PATH = "/cluster/huanglab/sluo/sluoo/escreen/eScreen_training/datasets/TestData_20260509.randomCRE.4.pkl"
# The five cell lines of the eScreen benchmark ("P0" freeze).
P0_CELL_LINES = ['A375', 'HT29', 'HepG2', 'K562', 'hPSC']
# WTC11 is the iPSC line; it is merged into the hPSC label so that all
# pluripotent samples share one cell-type id.
P0_RELABEL = {'WTC11': 'hPSC'}
# chr11 LCR / validation locus held out of train/valid/test. Note this is
# 5.20-5.30 Mb, wider than the 5.25-5.30 Mb range quoted in §2.3.3.
LOCUS_CHR = "chr11"
LOCUS_START = 5_200_000
LOCUS_END = 5_300_000
# Negative-downsampling rule (§2.3.4): a non-functional CRE counts as an
# unambiguous negative only below this RRA, and only if it is negative in at
# least MIN_CELL_LINES cell lines.
RRA_BOTTOM = 0.5
MIN_CELL_LINES = 2
# Fixed seed so the exported 7:2:1 split is reproducible.
SPLIT_SEED = 42

OUTPUT_PATH = "./enformer_embeddings/p0_filtered_samples.csv"

class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that rewrites ``numpy._core`` module paths to ``numpy.core``.

    Files pickled under numpy >= 2.0 reference ``numpy._core.*``, which does not
    exist in numpy 1.x (and vice versa). Rather than pinning a numpy version, the
    module path is rewritten on load so the same pickle can be read in either
    environment.

    Parameters
    ----------
    Inherits ``pickle.Unpickler``'s constructor (file-like object).

    Notes
    -----
    Only the module *prefix* is rewritten; the class name is passed through
    unchanged. This is safe because numpy kept a compatibility alias for
    ``numpy.core``.
    """

    def find_class(self, module, name):
        """Resolve a pickled class, normalising the numpy module prefix.

        Parameters
        ----------
        module : str
            Dotted module path recorded in the pickle.
        name : str
            Class (or function) name within that module.

        Returns
        -------
        type or function
            The resolved object.

        Notes
        -----
        Only ``numpy._core`` (numpy >= 2.0) is rewritten; every other module is
        delegated to the base implementation unchanged.
        """
        if module.startswith('numpy._core'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)

# ── Load original data ──
print('[Data] Loading original 8-tuple...')
with open(ORIGINAL_PATH, 'rb') as f:
    bundle = _NumpyCompatUnpickler(f).load()

trainset_o, validset_o, testset_o, train_idx, valid_idx, test_idx, cre_sub, final_df = bundle[:8]
print(f'[Data] Original: train={trainset_o["length"]}, valid={validset_o["length"]}, test={testset_o["length"]}')

# Flat list of every CRE-table row used by the three partitions. The *_idx
# arrays are *positions* into cre_sub, not positional ranges, so they are
# concatenated to form the candidate pool.
pool_index = list(train_idx) + list(valid_idx) + list(test_idx)

# ── Build lookup ──
def _build_lookup(t_idx, v_idx, te_idx):
    """Map each CRE-table row index to ``(split_name, position_within_split)``.

    Parameters
    ----------
    t_idx, v_idx, te_idx : sequence of int
        Row positions into ``cre_sub`` for the train, valid and test partitions
        of the original bundle.

    Returns
    -------
    dict
        ``{int(row_index): (split_name, j)}`` where ``j`` is the position inside
        that partition — i.e. the row's index in the corresponding tensor.

    Notes
    -----
    The result is assigned to ``lookup`` below but never subsequently used: the
    script re-splits the data from scratch instead of preserving the original
    partitioning (only the *pool* of rows is reused). Kept here because it
    documents how the original bundle is indexed.
    """
    lookup = {}
    for j, idx in enumerate(t_idx):
        lookup[int(idx)] = ('train', j)
    for j, idx in enumerate(v_idx):
        lookup[int(idx)] = ('valid', j)
    for j, idx in enumerate(te_idx):
        lookup[int(idx)] = ('test', j)
    return lookup

lookup = _build_lookup(train_idx, valid_idx, test_idx)

# ── Filter: keep positives + bottom negatives shared >=2 cell lines ──
# **_pool_idx** re-numbers the pooled rows 0..N-1 and is used as a positional
# handle later: after boolean filtering the DataFrame index no longer tracks the
# pool, so this column is the only reliable way back to `pool_index`.
meta = cre_sub.loc[pool_index].copy()
meta['_pool_idx'] = np.arange(len(pool_index))

# A "bottom negative": labelled 0 and with RRA below the 0.5 cut. The RRA test
# is what removes ambiguous negatives — a CRE with mediocre RRA might simply not
# have been assayed well, whereas RRA < 0.5 is strong evidence of no activity.
is_bottom_neg = (meta['label'] == 0) & (meta['RRA'] < RRA_BOTTOM)
# Keep only CRE ids that are bottom negatives in >= 2 cell lines: cross-cell-line
# agreement is the manuscript's criterion for an unambiguous negative (§2.3.4).
shared_ids = (
    meta.loc[is_bottom_neg]
    .groupby('id')['cell_line']
    .nunique()
    .pipe(lambda s: s[s >= MIN_CELL_LINES].index)
)
keep_mask = (meta['label'] == 1) | (is_bottom_neg & meta['id'].isin(shared_ids))

# ── Locus holdout ──
# Interval overlap test on 0-based half-open coordinates: a CRE is in the locus
# if it starts before the locus ends and ends after the locus starts. Using
# `start < END and end > START` (rather than containment) catches CREs that
# straddle either boundary.
locus_mask = (
    (meta['chr'] == LOCUS_CHR)
    & (meta['start'] < LOCUS_END)
    & (meta['end'] > LOCUS_START)
)
keep_split = keep_mask & ~locus_mask

# ── id-level 7:2:1 split ──
# The unit of splitting is the unique CRE id, not the row, so all cell-line rows
# of one CRE stay together (§2.3.3). Sizes are floored ints, so the realised
# ratio is 7:2:1 up to rounding and the last 10% absorbs the remainder.
split_meta = meta.loc[keep_split]
ids_arr = split_meta['id'].unique()
rng = np.random.default_rng(SPLIT_SEED)
rng.shuffle(ids_arr)
n_id = len(ids_arr)
n_train_id = int(0.7 * n_id)
n_valid_id = int(0.2 * n_id)
train_ids_set = set(ids_arr[:n_train_id])
valid_ids_set = set(ids_arr[n_train_id:n_train_id + n_valid_id])
test_ids_set = set(ids_arr[n_train_id + n_valid_id:])

train_rows = split_meta['id'].isin(train_ids_set)
valid_rows = split_meta['id'].isin(valid_ids_set)
test_rows = split_meta['id'].isin(test_ids_set)

# ── P0 filter ──
def _is_p0(idx):
    """Return True if the CRE-table row belongs to one of the five P0 cell lines.

    Parameters
    ----------
    idx : int
        *Label* index into ``cre_sub`` (not a position in the filtered pool).

    Returns
    -------
    bool
        True for A375 / HT29 / HepG2 / K562 / hPSC and for WTC11 (which is
        relabelled to hPSC by the rest of the pipeline).

    Notes
    -----
    Uses ``cre_sub.loc[idx, 'cell_line']`` — label-based lookup — so it must be
    called with values from the original index, exactly as ``pool_index`` holds
    them. The WTC11 branch returns whether 'hPSC' appears among the relabel
    targets, which is a constant True in practice; it is written that way to keep
    the relabel mapping as the single source of truth.
    """
    cl = str(cre_sub.loc[idx, 'cell_line'])
    if cl.upper() == 'WTC11':
        return 'hPSC' in P0_RELABEL.values()
    return cl in set(P0_CELL_LINES)

# Collect indices per split
# split_name → list of original cre_sub label indices. Note `_pool_idx` values
# are positions in the *pool*, so they must be mapped back through pool_index
# before being used with .loc.
split_data = {}  # split_name → list of original indices
for name, row_set in [('train', train_rows), ('valid', valid_rows), ('test', test_rows)]:
    _split_idx = split_meta.loc[row_set, '_pool_idx'].astype(int).tolist()
    filtered = [pool_index[i] for i in _split_idx if _is_p0(pool_index[i])]
    split_data[name] = filtered

# Locus
# The locus rows were excluded from the 7:2:1 draw but are kept as a fourth,
# fully held-out split so that locus-specific performance can still be reported.
_locus_raw = meta.loc[keep_mask & locus_mask, '_pool_idx'].astype(int).tolist()
locus_filtered = [pool_index[i] for i in _locus_raw if _is_p0(pool_index[i])]
split_data['locus'] = locus_filtered

total = sum(len(v) for v in split_data.values())
print(f'[Data] P0-filtered: train={len(split_data["train"])}, valid={len(split_data["valid"])}, test={len(split_data["test"])}, locus={len(split_data["locus"])} (total={total})')

# ── Build output DataFrame ──
# `subset = cre_sub.loc[indices]` keeps the original column set and dtypes; the
# two bookkeeping columns are prepended at the end for readability.
rows_list = []
for split_name, indices in split_data.items():
    subset = cre_sub.loc[indices].copy()
    subset['split'] = split_name
    subset['original_index'] = indices
    rows_list.append(subset)

df = pd.concat(rows_list)
# Reorder columns: put split and original_index first, then original columns
orig_cols = [c for c in cre_sub.columns if c not in ('split', 'original_index')]
df = df[['split', 'original_index'] + orig_cols]
df = df.reset_index(drop=True)

print(f'[Data] Output shape: {df.shape}')
print(f'[Data] Split counts:')
print(df['split'].value_counts())

# Save
# The CSV is the interchange format used by extraEnformerEmb.py; the pickle is
# written for callers that need the dtypes back (labels as int, not str).
df.to_csv(OUTPUT_PATH, index=False)
print(f'[Data] Saved to {OUTPUT_PATH}')

# Also save as pickle for easy loading
pkl_path = OUTPUT_PATH.replace('.csv', '.pkl')
df.to_pickle(pkl_path)
print(f'[Data] Also saved to {pkl_path}')
