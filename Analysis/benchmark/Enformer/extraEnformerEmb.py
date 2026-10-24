#!/usr/bin/env python3
"""
Extract Enformer trunk embeddings (CRE bins → [B, 4, 3072])
for P0-filtered CREs using PATCH-BASED splitting and embedding extraction.

Key differences from original per-CRE approach:
  1. Patch-based split: divide each chromosome into 114688bp patches,
     assign CREs to patches, group consecutive patches into blocks,
     then split blocks into train/val/test at block level (7:2:1 CRE ratio).
  2. One Enformer inference per patch: compute trunk embedding once per patch
     (896 bins × 3072 ch), then map each CRE in the patch to its 4
     corresponding bins (left-aligned; if CRE spans 5 bins, drop the rightmost).
  3. Single GPU (cuda:1).

Enformer layout (896 bins, 128 bp/bin):
  ┌────────────── 393216 bp input ─────────────────┐
  │  139264 flank  │██ 114688 target ██│ 139264 flank │
                     └── 896 bins × 128bp ──┘
  Patch (114688bp) aligns exactly with the target region.

Output (in {output_dir}/):
  cre_info.npy          - structured array (cre_id, chr, start, end, emb_idx)
  cre_to_emb_idx.pkl    - dict: cre_id -> embedding row index
  sample_cre_ids.npy    - CRE id per P0-filtered sample (for downstream lookup)
  sample_split.npy      - split label per sample (0=train, 1=valid, 2=test)
  unique_cre_split.npy  - split label per unique CRE (same order as cre_info)
  embeddings.memmap     - float32 memmap (N_unique, 4, 3072)

What this script implements
---------------------------
The Enformer embedding extraction used for the "genomic block embedding
finetune" strategy of Supplementary Methods §6.2.1. The paper's intent is: take
a DNA window around each CRE, run Enformer once, and keep the trunk (pre-output-
head) activation at the bins covering the CRE as a fixed feature vector, which a
small MLP head then maps to the CRE-activity label (see
``train_enformer_emb_new.py``).

Rather than one Enformer call per CRE, this implementation makes one call per
114,688 bp *patch* and reuses the result for every CRE that falls in it — hence
"patch-based". That is a throughput optimisation, and it also makes the
train/valid/test split spatial: CREs that share a patch necessarily share a
split, which removes the train/test leakage that a per-CRE split would allow for
neighbouring elements.

Where the code and the manuscript disagree
------------------------------------------
§6.2.1 states that the CRE-centred variant extends 131,072 bp on each side and the
genomic-block variant 262,144 bp on each side. Neither number appears here: the
code uses Enformer's native receptive field, 393,216 bp total, arranged as
``139,264 | 114,688 | 139,264`` so that the patch is exactly the model's centred
bin grid. The manuscript's numbers are not reachable with this bin geometry, so
the paper's flank sizes describe a different (earlier) run.

Also note that this script does **not** read the P0-filtered CSV produced by
``export_p0_samples.py``. It re-derives the same sample filter from the original
pickle (``RRA < 0.5``, shared across >= 2 cell lines, the five P0 lines) and then
replaces the 7:2:1 identifier-level split with its own patch/block split. The two
ways of building the benchmark are therefore independent, and the sample counts
can diverge silently.

Expected on disk
----------------
Inputs (all hard-coded absolute paths on the authors' cluster)::

    /cluster/huanglab/sluo/sluoo/escreen/eScreen_training/datasets/
        TestData_20260509.randomCRE.4.pkl      # 8-tuple eScreen bundle
    /cluster2/huanglab/liquan/genome_ref/hg38.fa            # pyfaidx FASTA
    /cluster2/huanglab/liquan/data/eSCREEN/ANLYSIS/code/Enformer/enformer_model
        # TensorFlow SavedModel of Enformer (tf.keras.models.load_model)

Outputs, in ``{script_dir}/enformer_embeddings/``::

    cre_info.npy          structured (cre_id, chr, start, end, emb_idx)
    cre_to_emb_idx.pkl    dict cre_id -> memmap row
    sample_cre_ids.npy    CRE id per P0-filtered sample
    sample_split.npy      int8 split per sample: 0=train, 1=valid, 2=test
    unique_cre_split.npy  int8 split per unique CRE (same order as cre_info)
    embeddings.memmap     float32 (n_unique, 4, 3072), ~5.9 GB for 122k CREs

Usage::

    python extraEnformerEmb.py                  # main process + one worker
    python extraEnformerEmb.py --worker --gpu 1 \
        --cre_info ... --memmap ... --n_total ...

The main process computes the split and pre-allocates the memmap, then shells out
to itself with ``--worker``; a single worker on GPU 1 fills the shared memmap.
The subprocess design keeps TensorFlow out of the coordinating process, which
otherwise holds the whole embedding dict and the FASTA handle.

External dependencies not vendored in this repository: ``tensorflow`` (2.x),
``pyfaidx``, and the Enformer SavedModel itself. The Enformer implementation is
*not* part of this repo — it is the ``enformer_model`` directory produced by the
upstream ``enformer``/``basenji`` project.
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths & Constants ─────────────────────────────────────
ORIGINAL_PATH = "/cluster/huanglab/sluo/sluoo/escreen/eScreen_training/datasets/TestData_20260509.randomCRE.4.pkl"
FASTA_PATH = "/cluster2/huanglab/liquan/genome_ref/hg38.fa"
ENFORMER_MODEL_PATH = "/cluster2/huanglab/liquan/data/eSCREEN/ANLYSIS/code/Enformer/enformer_model"

P0_CELL_LINES = ['A375', 'HT29', 'HepG2', 'K562', 'hPSC']
P0_RELABEL = {'WTC11': 'hPSC'}
RRA_BOTTOM = 0.5
MIN_CELL_LINES = 2
SPLIT_SEED = 42

SEQUENCE_LENGTH = 393216
PATCH_SIZE = 114688
FLANK = (SEQUENCE_LENGTH - PATCH_SIZE) // 2          # 139264 flank on each side
BIN_WIDTH = 128                                       # bp per Enformer trunk bin
PATCH_BINS = PATCH_SIZE // BIN_WIDTH                  # 896
TRUNK_CHANNELS = 3072
CRE_BIN_COUNT = 4
BATCH_SIZE = 4

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'enformer_embeddings')

SPLIT_LABELS = {'train': 0, 'valid': 1, 'test': 2}


# ── Numpy compat (old pickle format) ────────────────────
class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that rewrites ``numpy._core`` module paths to ``numpy.core``.

    The original eScreen bundle was pickled under a different numpy major version
    than the one used for extraction. numpy >= 2.0 moved the C-extension modules
    to ``numpy._core``; rewriting the prefix on load lets the same pickle be read
    in either environment without pinning numpy.

    Parameters
    ----------
    Inherits ``pickle.Unpickler``'s constructor (a file-like object).

    Notes
    -----
    Only the module prefix is rewritten; numpy kept ``numpy.core`` as a
    compatibility alias, so the resolved classes are identical.
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


# ══════════════════════════════════════════════════════════
#  Step 1 — Data Loading & P0 Filtering
# ══════════════════════════════════════════════════════════

def load_original_data():
    """Load the original 8-tuple pickle.

    Returns
    -------
    tuple
        The first eight elements of the bundle:
        ``(trainset, validset, testset, train_idx, valid_idx, test_idx,
        cre_sub, final_df)``. Only ``*_idx`` (positions into ``cre_sub``) and
        ``cre_sub`` itself are used downstream; the three dataset dicts are read
        for their ``"length"`` keys in the log line only.

    Notes
    -----
    Uses :class:`_NumpyCompatUnpickler` so the file loads under both numpy 1.x
    and 2.x. Slice ``[:8]`` is defensive: older exports carried extra trailing
    objects.
    """
    print(f'[Data] Loading original 8-tuple from {ORIGINAL_PATH}')
    with open(ORIGINAL_PATH, 'rb') as f:
        bundle = _NumpyCompatUnpickler(f).load()
    return bundle[:8]


def get_unique_cre_data():
    """
    Full pipeline: load → filter → P0 → dedup by CRE id.
    No id-level train/val/test split (replaced by patch-level split later).

    Parameters
    ----------
    None
        All paths come from module-level constants (hard-coded cluster paths).

    Returns
    -------
    unique_cre : pandas.DataFrame
        Columns ``id``, ``chr``, ``start``, ``end``; one row per unique CRE id,
        ordered by first appearance so that row *i* is embedding row *i*.
    cre_to_emb : dict of str to int
        ``cre_id -> row index`` into the embedding memmap.
    sample_cre_ids : numpy.ndarray
        CRE id of every P0-filtered *sample* (a CRE can appear once per cell
        line), in the same order as ``p0_meta``.
    p0_meta : pandas.DataFrame
        The filtered rows of ``cre_sub`` including ``label``, ``RRA`` and
        ``cell_line``.

    Notes
    -----
    Shorthand for the four return values, as originally written::

        unique_cre      — DataFrame (id, chr, start, end) one row per unique CRE
        cre_to_emb      — dict  cre_id → embedding row index
        sample_cre_ids  — ndarray  CRE id of each P0-filtered sample
        p0_meta         — DataFrame of all P0-filtered samples

    * "Deduplicate by CRE id" keeps the first row per id, so ``start``/``end`` are
      taken from whichever cell line happened to come first. The interval is
      identical across cell lines in this dataset, so this is safe here, but it
      is an identity assumption, not a checked one.
    * No train/valid/test split happens here; that is
      :func:`create_patch_split`'s job. This function only defines the pool.
    * ``p0_meta`` still carries the original ``cre_sub`` index, which
      ``train_enformer_emb_new.py`` relies on when it re-derives the same rows.
    """
    # 1. Load
    trainset_o, validset_o, testset_o, train_idx, valid_idx, test_idx, cre_sub, final_df = \
        load_original_data()
    print(f'[Data] Original: train={trainset_o["length"]}, '
          f'valid={validset_o["length"]}, test={testset_o["length"]}')

    pool_index = list(train_idx) + list(valid_idx) + list(test_idx)

    # 2. Filter: keep positives + bottom negatives shared across >=2 cell lines
    meta = cre_sub.loc[pool_index].copy()
    meta['_pool_idx'] = np.arange(len(pool_index))

    is_bottom_neg = (meta['label'] == 0) & (meta['RRA'] < RRA_BOTTOM)
    shared_ids = (
        meta.loc[is_bottom_neg]
        .groupby('id')['cell_line']
        .nunique()
        .pipe(lambda s: s[s >= MIN_CELL_LINES].index)
    )
    keep_mask = (meta['label'] == 1) | (is_bottom_neg & meta['id'].isin(shared_ids))

    # 3. P0 filter
    def _is_p0(idx):
        """Return True if a CRE-table row belongs to one of the five P0 cell lines.

        Parameters
        ----------
        idx : int
            *Label* index into ``cre_sub`` (a value taken from ``pool_index``).

        Returns
        -------
        bool
            True for A375 / HT29 / HepG2 / K562 / hPSC, and for WTC11, which the
            benchmark relabels as hPSC.

        Notes
        -----
        The WTC11 branch tests ``'hPSC' in P0_RELABEL.values()`` — a constant
        True — rather than hard-coding ``True``, so that removing the relabel
        entry would also disable the WTC11 inclusion. ``str()`` guards against
        NaN cell-line values.
        """
        cl = str(cre_sub.loc[idx, 'cell_line'])
        if cl.upper() == 'WTC11':
            return 'hPSC' in P0_RELABEL.values()
        return cl in set(P0_CELL_LINES)

    keep_indices = meta.loc[keep_mask, '_pool_idx'].astype(int).tolist()
    p0_filtered = [pool_index[i] for i in keep_indices if _is_p0(pool_index[i])]

    print(f'[Data] P0-filtered total samples: {len(p0_filtered)}')

    # 4. Dedup by CRE id
    p0_meta = cre_sub.loc[p0_filtered]
    unique_cre = p0_meta[['id', 'chr', 'start', 'end']].drop_duplicates(subset='id')
    unique_cre = unique_cre.reset_index(drop=True)
    n_unique = len(unique_cre)
    print(f'[Data] Unique CREs after dedup: {n_unique}')

    cre_to_emb = {row['id']: i for i, row in unique_cre.iterrows()}
    sample_cre_ids = p0_meta['id'].values

    return unique_cre, cre_to_emb, sample_cre_ids, p0_meta


# ══════════════════════════════════════════════════════════
#  Step 2 — Patch-based Train/Val/Test Split
# ══════════════════════════════════════════════════════════

def create_patch_split(unique_cre, seed=SPLIT_SEED):
    """
    Patch-level 7:2:1 split.

    1. Assign each unique CRE to its 114688bp patch per chromosome.
    2. Group consecutive patches on each chromosome into blocks.
       Block CRE cap ≈ 5% of total for good splitting granularity.
    3. Greedy-assign each block to train/val/test using deficit maximisation.

    Parameters
    ----------
    unique_cre : pandas.DataFrame
        Output of :func:`get_unique_cre_data`; needs ``id``, ``chr``, ``start``.
    seed : int, optional
        Seed for the block-order shuffle. Defaults to :data:`SPLIT_SEED` (42),
        matching the identifier-level split used elsewhere in the benchmark.

    Returns
    -------
    cre_to_split : dict of str to str
        ``cre_id -> 'train' | 'valid' | 'test'``.
    split_counts : dict of str to int
        Realised CRE counts per split (not exactly 70/20/10 — see Notes).

    Notes
    -----
    Shorthand for the two return values, as originally written::

        cre_to_split  — dict  cre_id → 'train'|'valid'|'test'
        split_counts  — dict  split_name → int count

    Three levels of granularity, from finest to coarsest:

    1. **Patch** — a fixed 114,688 bp tile, ``start // PATCH_SIZE``. Each patch
       maps 1:1 onto Enformer's 896 output bins, so one inference serves every
       CRE in the patch.
    2. **Block** — consecutive patches on one chromosome, greedily accumulated
       until adding another patch would exceed ``5%`` of all CREs. The cap trades
       split granularity against block count; it is a heuristic, not a manuscript
       parameter.
    3. **Split** — blocks are shuffled with ``seed`` and sorted large-first, then
       each block goes to whichever split has the largest remaining deficit
       (``target - current``). This greedy "deficit maximisation" gets close to
       7:2:1 without ever splitting a block.

    Consequences worth knowing:

    * The realised ratios are approximate and depend on block sizes; a single
      huge block can overshoot a target by several percent.
    * Because assignment is spatial, a chromosome can contribute to all three
      splits. Leakage across splits is prevented at the patch level, not the
      chromosome level, unlike the chromosome-held-out scheme in Supplementary
      Methods §2.3.3. CREs in *adjacent* patches can therefore still share
      long-range regulatory context.
    * Patch membership is decided by the CRE *start* only, so a CRE straddling a
      patch boundary is assigned to the patch containing its start; the 4
      embedded bins are then clamped inside the patch in :func:`worker`.
    """
    total = len(unique_cre)

    # ── 1. Assign CREs to patches ──
    patches = defaultdict(list)          # (chrom, patch_idx) → [cre_id]
    for _, row in unique_cre.iterrows():
        chrom = row['chr']
        patch_idx = int(row['start']) // PATCH_SIZE
        patches[(chrom, patch_idx)].append(row['id'])

    # ── 2. Organise by chromosome, sorted by patch index ──
    chrom_data = defaultdict(list)       # chrom → [(patch_idx, [cre_id])]
    for (chrom, pidx), cre_ids in patches.items():
        chrom_data[chrom].append((pidx, cre_ids))
    for chrom in chrom_data:
        chrom_data[chrom].sort(key=lambda x: x[0])

    # ── 3. Form blocks by merging consecutive patches ──
    max_block_cre = max(1, int(total * 0.05))   # ~5% cap for fine granularity
    blocks = []                                   # each: (chrom, [patch_idx], cre_count)
    for chrom in sorted(chrom_data.keys()):
        patch_list = chrom_data[chrom]
        cur_patches = []
        cur_count = 0
        for pidx, cres in patch_list:
            nc = len(cres)
            if cur_count + nc > max_block_cre and cur_patches:
                blocks.append((chrom, list(cur_patches), cur_count))
                cur_patches = [pidx]
                cur_count = nc
            else:
                cur_patches.append(pidx)
                cur_count += nc
        if cur_patches:
            blocks.append((chrom, list(cur_patches), cur_count))

    # ── 4. Greedy assignment to train / valid / test ──
    rng = np.random.default_rng(seed)
    order = list(range(len(blocks)))
    rng.shuffle(order)
    order.sort(key=lambda i: blocks[i][2], reverse=True)   # large blocks first

    targets = {'train': total * 0.7, 'valid': total * 0.2, 'test': total * 0.1}
    current = {'train': 0, 'valid': 0, 'test': 0}
    block_split = {}

    for idx in order:
        bc = blocks[idx][2]
        deficit = {s: targets[s] - current[s] for s in targets}
        best = max(deficit, key=deficit.get)
        block_split[idx] = best
        current[best] += bc

    # ── 5. Build cre_id → split mapping ──
    cre_to_split = {}
    for idx, (chrom, pindices, _) in enumerate(blocks):
        split_label = block_split[idx]
        for pidx in pindices:
            for cre_id in patches[(chrom, pidx)]:
                cre_to_split[cre_id] = split_label

    # ── Summary ──
    print(f'[Split] Patch-level 7:2:1 split:')
    print(f'[Split]   Total CREs: {total}')
    print(f'[Split]   Blocks: {len(blocks)}')
    print(f'[Split]   Patches with CREs: {len(patches)}')
    for s in ['train', 'valid', 'test']:
        pct = current[s] / total * 100
        print(f'[Split]   {s.capitalize()}: {current[s]} ({pct:.1f}%)')

    return cre_to_split, current


# ══════════════════════════════════════════════════════════
#  Step 3 — Sequence Extraction
# ══════════════════════════════════════════════════════════

def one_hot_encode_dna(sequence, dtype=np.float32):
    """One-hot encode DNA string via uint8 lookup table.

    Parameters
    ----------
    sequence : str
        DNA sequence, any case; nucleotides other than A/C/G/T are treated as
        unknown.
    dtype : numpy.dtype, optional
        Output dtype, ``float32`` by default because that is what Enformer's
        input placeholder expects.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(sequence), 4)``, channel order ``A, C, G, T``. Rows for
        ``N`` (and any other non-ACGT byte) are all zeros.

    Notes
    -----
    * The lookup table is indexed directly by the ASCII byte of each character,
      which is much faster than a per-base dict or ``np.eye`` gather on 393 kb
      sequences — this function is called once per patch, i.e. ~1.6 M bases per
      batch of 4.
    * ``table`` has only 255 rows (``np.iinfo(np.uint8).max``), so any byte
      >= 255 would index out of range; DNA strings (plus ``N``) never reach that,
      but a stray non-ASCII character would raise rather than be silently
      zeroed.
    * Row 0 of the table stays all-zero, which is what makes unmapped characters
      encode as "unknown" for free.
    * ``.upper()`` is applied before encoding so soft-masked (lowercase) FASTA
      does not silently become all-zero.
    """
    alphabet = 'ACGT'
    neutral = 'N'

    def _to_uint8(s):
        """View a string's ASCII bytes as a ``uint8`` array (no copy of characters).

        Parameters
        ----------
        s : str
            ASCII string, here either the alphabet or a full DNA sequence.

        Returns
        -------
        numpy.ndarray
            ``uint8`` array of the same length, one byte per character — usable
            directly as a fancy-index into the lookup table.

        Notes
        -----
        ``np.frombuffer`` on ``s.encode('ascii')`` is the fast path: it avoids
        Python-level iteration over ~393 k characters per sequence. Encoding as
        ASCII (rather than UTF-8) guarantees one byte per character, which is what
        makes the byte values usable as table indices.
        """
        return np.frombuffer(s.encode('ascii'), dtype=np.uint8)

    table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    table[_to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    table[_to_uint8(neutral)] = 0.0
    return table[_to_uint8(sequence.upper())]


def extract_sequence(fasta, chrom, start_0based, end_0based):
    """
    Extract DNA sequence from reference genome (pyfaidx).
    Pads with N at chromosome boundaries.

    Parameters
    ----------
    fasta : pyfaidx.Fasta
        Indexed hg38.
    chrom : str
        Chromosome name, e.g. ``'chr7'``.
    start_0based, end_0based : int
        Requested interval, 0-based half-open. Values outside the chromosome are
        allowed; the out-of-range part becomes ``N``.

    Returns
    -------
    str
        Sequence of length ``end_0based - start_0based`` (always exact, because
        the missing flanks are padded).

    Notes
    -----
    * **Coordinate conversion**: pyfaidx's ``get_seq`` takes 1-based *inclusive*
      bounds, while everything in this pipeline is 0-based half-open. Hence
      ``trimmed_start + 1`` on the left and plain ``trimmed_end`` on the right —
      the off-by-one is intentional, not a bug.
    * ``max(-start_0based, 0)`` / ``max(end_0based - chrom_length, 0)`` compute
      the padding on each side; both are 0 when the window is fully inside the
      chromosome.
    * The padding keeps the returned length constant, which matters because the
      caller stacks sequences into a fixed ``(B, 393216, 4)`` array and because
      the patch-to-bin mapping assumes the window start is ``patch_start -
      FLANK`` regardless of clipping.
    """
    chrom_length = len(fasta[chrom])
    trimmed_start = max(start_0based, 0)
    trimmed_end = min(end_0based, chrom_length)
    seq = str(fasta.get_seq(chrom, trimmed_start + 1, trimmed_end)).upper()
    pad_up = 'N' * max(-start_0based, 0)
    pad_down = 'N' * max(end_0based - chrom_length, 0)
    return pad_up + seq + pad_down


# ══════════════════════════════════════════════════════════
#  Worker — Patch-based Enformer Inference (single GPU)
# ══════════════════════════════════════════════════════════

def worker(gpu_id, cre_info_path, memmap_path, n_total):
    """
    Spawned as subprocess: load Enformer on assigned GPU,
    process patches, write CRE embeddings to shared memmap.

    Each patch (114688bp) gets ONE Enformer inference.  All CREs in that
    patch share the result.  For each CRE we determine its 4 output bins
    (left-aligned) and extract the corresponding embedding slices.

    Parameters
    ----------
    gpu_id : int
        Physical GPU index, exported as ``CUDA_VISIBLE_DEVICES`` before
        TensorFlow is imported (import order matters — TF must not be imported
        by the parent process first).
    cre_info_path : str
        ``cre_info.npy`` written by :func:`main`: structured array with fields
        ``cre_id``, ``chr``, ``start``, ``end``, ``emb_idx``.
    memmap_path : str
        Pre-allocated ``float32`` memmap, ``(n_total, 4, 3072)``, opened in
        ``r+`` mode and written in place.
    n_total : int
        Number of unique CREs, i.e. the memmap's first dimension.

    Returns
    -------
    None
        Results are written into the memmap; the parent process reads them after
        the subprocess exits.

    Notes
    -----
    * **Graph surgery.** The SavedModel is a full Enformer (trunk + heads). To get
      the trunk embedding, the code traces the concrete function, digs the inner
      ``StatefulPartitionedCall`` graph out of it, re-imports that graph, and
      prunes it to the single tensor ``seqnn/trunk/final_pointwise/mul_1:0``.
      That tensor name is Enformer-internal and will break if the upstream model
      is re-exported with different layer names. The pruned function is then
      called with the sequence plus the concrete function's captured inputs.
    * **Bin mapping.** The 114,688 bp patch corresponds exactly to the middle 896
      of Enformer's 896 bins (1 bin = 128 bp), because the patch is centred
      inside the 393,216 bp input. For a CRE, ``rel_start = cstart -
      patch_start`` and ``b_start = rel_start // BIN_WIDTH``; four consecutive bins
      are taken left-aligned. ``4 bins * 128 bp = 512 bp``, matching the eScreen
      input length, so the embedding covers the same footprint as the sequence the
      other models see.
    * **Spans-5-bins case.** A CRE whose 512 bp window is not 128 bp aligned can
      cover five bins; the original per-CRE approach dropped the rightmost bin,
      and this code instead clamps ``b_start`` into
      ``[0, PATCH_BINS - 4]``, which has the same effect at the patch interior and
      additionally prevents a negative or past-the-end slice at patch edges.
    * `processed % 5000 == 0` is tested against the cumulative CRE counter inside
      the per-patch loop, so the progress line can print several times for one
      batch or be skipped entirely depending on how many CREs each patch holds.
    * Batches are formed from a plain ``list(patch_groups.items())``; dict order is
      insertion order, i.e. CRE order in ``cre_info``. There is no shuffling and no
      resume: a crash restarts from the first patch, and the memmap is partially
      filled.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    import pyfaidx

    # ── Load Enformer & prune graph for trunk embedding ──
    print(f'[Worker] Loading Enformer model...')
    weight = tf.keras.models.load_model(ENFORMER_MODEL_PATH)

    concrete = weight.model.predict_on_batch.get_concrete_function(
        tf.TensorSpec([None, SEQUENCE_LENGTH, 4], tf.float32)
    )
    graph = concrete.graph
    op = graph.get_operation_by_name("StatefulPartitionedCall")
    func_name = op.get_attr("f").name
    atomic_fn = graph._functions[func_name]
    inner_graph = atomic_fn.graph

    graph_def = inner_graph.as_graph_def()
    wrapped = tf.compat.v1.wrap_function(
        lambda: tf.graph_util.import_graph_def(graph_def, name=""), []
    )

    all_phs = [op.outputs[0] for op in wrapped.graph.get_operations()
               if op.type == "Placeholder"]

    embedding_tensor = wrapped.graph.get_tensor_by_name(
        "seqnn/trunk/final_pointwise/mul_1:0"
    )

    embedding_fn = wrapped.prune(feeds=all_phs, fetches=[embedding_tensor])
    captured_inputs = list(concrete._captured_inputs)

    def get_embedding(seq_input):
        """Run the pruned Enformer trunk on one batch of sequences.

        Parameters
        ----------
        seq_input : numpy.ndarray or torch.Tensor
            One-hot DNA of shape ``(B, 393216, 4)``.

        Returns
        -------
        tensor
            Trunk activations of shape ``(B, 896, 3072)`` — 896 bins of 128 bp,
            3072 channels, i.e. the layer before Enformer's output heads.

        Notes
        -----
        The pruned concrete function expects the sequence followed by the
        original concrete function's captured inputs, hence
        ``embedding_fn(seq_input, *captured_inputs)``. The
        ``isinstance(result, (list, tuple))`` guard handles both a single-tensor
        and a multi-output fetch; only the first (and only) fetch is configured, so
        either way one tensor comes back.
        """
        seq_input = tf.convert_to_tensor(seq_input, dtype=tf.float32)
        result = embedding_fn(seq_input, *captured_inputs)
        return result[0] if isinstance(result, (list, tuple)) else result

    # ── Open reference genome ──
    print(f'[Worker] Loading reference genome...')
    fasta = pyfaidx.Fasta(FASTA_PATH)

    # ── Open shared memmap ──
    emb = np.memmap(memmap_path, dtype=np.float32, mode='r+',
                    shape=(n_total, CRE_BIN_COUNT, TRUNK_CHANNELS))

    # ── Load CRE info & group by patch ──
    cre_info = np.load(cre_info_path, allow_pickle=True)
    # cre_info has fields: cre_id, chr, start, end, emb_idx

    # Build patch groups
    patch_groups = {}   # (chrom, patch_idx) → {patch_start, win_start, win_end, cres}
    for cre in cre_info:
        chrom = str(cre['chr'])
        cstart = int(cre['start'])
        emb_idx = int(cre['emb_idx'])
        patch_idx = cstart // PATCH_SIZE
        key = (chrom, patch_idx)
        if key not in patch_groups:
            pstart = patch_idx * PATCH_SIZE
            patch_groups[key] = {
                'patch_start': pstart,
                'win_start': pstart - FLANK,
                'win_end': pstart + PATCH_SIZE + FLANK,
                'cres': [],      # (emb_idx, cstart)
            }
        patch_groups[key]['cres'].append((emb_idx, cstart))

    patch_items = list(patch_groups.items())
    n_patches = len(patch_items)
    n_cre_total = len(cre_info)
    print(f'[Worker] Processing {n_patches} patches ({n_cre_total} CREs)...')

    # ── Process patches in batches ──
    t0 = time.time()
    processed = 0

    for batch_start in range(0, n_patches, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_patches)
        batch = patch_items[batch_start:batch_end]
        bs = batch_end - batch_start

        # Extract & one-hot encode sequences for all patches in batch
        batch_seqs = []
        for key, info in batch:
            seq_str = extract_sequence(fasta, key[0],
                                       info['win_start'], info['win_end'])
            batch_seqs.append(one_hot_encode_dna(seq_str))

        batch_input = np.stack(batch_seqs, axis=0)  # (bs, 393216, 4)

        # Enformer inference → trunk embedding (bs, 896, 3072)
        batch_output = get_embedding(batch_input)

        # For each patch, extract CRE embeddings
        for i, (key, info) in enumerate(batch):
            patch_emb = batch_output[i]   # (896, 3072)
            for emb_idx, cstart in info['cres']:
                # Determine 4 bins for this CRE (left-aligned)
                rel_start = cstart - info['patch_start']
                b_start = rel_start // BIN_WIDTH
                b_start = max(0, min(b_start, PATCH_BINS - CRE_BIN_COUNT))
                emb[emb_idx] = patch_emb[b_start:b_start + CRE_BIN_COUNT, :]
                processed += 1

        # Progress logging
        if processed % 5000 == 0 or batch_end == n_patches:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            print(f'[Worker]  {batch_end}/{n_patches} patches, '
                  f'{processed}/{n_cre_total} CREs  ({rate:.0f} CREs/sec)')

    emb.flush()
    elapsed = time.time() - t0
    print(f'[Worker] DONE — {n_patches} patches, {processed} CREs '
          f'in {elapsed:.0f}s ({processed / elapsed:.1f} CREs/sec)')


# ══════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════

def main():
    """Coordinate the extraction: build metadata, allocate the memmap, run one worker.

    Returns
    -------
    None

    Notes
    -----
    Steps, in order (each has a side effect on disk):

    1. Rebuild the P0-filtered unique-CRE pool from the original pickle
       (:func:`get_unique_cre_data`).
    2. Split it spatially into train/valid/test (:func:`create_patch_split`) and
       map every *sample* to its split label. Samples inherit their CRE's split,
       so a CRE seen in three cell lines yields three rows with the same label.
    3. Write the metadata: ``cre_info.npy`` (unique CREs, with the memmap row
       index), ``cre_to_emb_idx.pkl``, ``sample_cre_ids.npy``, ``sample_split.npy``
       and ``unique_cre_split.npy``.
    4. Pre-allocate the ``float32`` memmap with ``mode='w+'`` — this truncates any
       previous extraction, so a re-run always starts from scratch.
    5. Spawn ``sys.executable __file__ --worker --gpu 1 ...`` and wait. GPU 1 is
       **hard-coded** in the command line; the ``--gpu`` default (0) is only used
       if the script is invoked in worker mode by hand.

    The parent blocks on ``p.wait()`` and propagates a non-zero worker exit code,
    so a TensorFlow failure surfaces as a script failure rather than as a silently
    half-filled memmap.

    ``sample_split`` is computed but only ever saved, never used in-process; it is
    the contract between this script and ``train_enformer_emb_new.py``.
    """
    parser = argparse.ArgumentParser(
        description='Extract Enformer trunk embeddings (patch-based)')
    parser.add_argument('--worker', action='store_true',
                        help='Worker subprocess mode')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cre_info', type=str)
    parser.add_argument('--memmap', type=str)
    parser.add_argument('--n_total', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    # ── Worker mode ──
    if args.worker:
        worker(args.gpu, args.cre_info, args.memmap, args.n_total)
        return

    # ── Main process ──
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print('=' * 60)
    print('  Enformer Embedding Extraction  (patch-based scheme)')
    print('=' * 60)

    t_start = time.time()

    # 1. Get unique CREs (P0-filtered, deduped)
    unique_cre, cre_to_emb, sample_cre_ids, p0_meta = get_unique_cre_data()
    n_unique = len(unique_cre)

    if n_unique == 0:
        print('[Main] ERROR: No unique CREs found. Exiting.')
        sys.exit(1)

    # 2. Patch-level train/val/test split
    cre_to_split, split_counts = create_patch_split(unique_cre)

    # 3. Assign each P0-filtered sample to a split
    sample_split = np.array(
        [SPLIT_LABELS[cre_to_split.get(cid, 'train')] for cid in sample_cre_ids],
        dtype=np.int8
    )

    # 4. Save metadata
    # 4a. cre_info with emb_idx (for worker lookup)
    cre_info_arr = np.array(
        [(row['id'], row['chr'], int(row['start']), int(row['end']), i)
         for i, (_, row) in enumerate(unique_cre.iterrows())],
        dtype=[('cre_id', 'U100'), ('chr', 'U10'),
               ('start', 'i4'), ('end', 'i4'), ('emb_idx', 'i4')]
    )

    cre_info_path = os.path.join(output_dir, 'cre_info.npy')
    np.save(cre_info_path, cre_info_arr)

    with open(os.path.join(output_dir, 'cre_to_emb_idx.pkl'), 'wb') as f:
        pickle.dump(cre_to_emb, f)

    np.save(os.path.join(output_dir, 'sample_cre_ids.npy'), sample_cre_ids)
    np.save(os.path.join(output_dir, 'sample_split.npy'), sample_split)

    # Per-unique-CRE split labels (0=train, 1=valid, 2=test)
    uc_split = np.array(
        [SPLIT_LABELS[cre_to_split.get(row['id'], 'train')]
         for _, row in unique_cre.iterrows()],
        dtype=np.int8
    )
    np.save(os.path.join(output_dir, 'unique_cre_split.npy'), uc_split)

    print(f'[Main] Metadata saved to {output_dir}')

    # 5. Pre-allocate memmap
    memmap_path = os.path.join(output_dir, 'embeddings.memmap')
    memmap_shape = (n_unique, CRE_BIN_COUNT, TRUNK_CHANNELS)
    _ = np.memmap(memmap_path, dtype=np.float32, mode='w+', shape=memmap_shape)
    file_size_gb = n_unique * CRE_BIN_COUNT * TRUNK_CHANNELS * 4 / 1e9
    print(f'[Main] Memmap allocated ({file_size_gb:.2f} GB): {memmap_path}')

    # 6. Spawn single worker on GPU 1 (cuda:1)
    print(f'[Main] Spawning worker on GPU 1 (cuda:1) for all {n_unique} CREs...')
    cmd = [
        sys.executable, __file__,
        '--worker', '--gpu', '1',
        '--cre_info', cre_info_path,
        '--memmap', memmap_path,
        '--n_total', str(n_unique),
    ]
    p = subprocess.Popen(cmd)
    p.wait()

    if p.returncode != 0:
        print(f'[Main] ERROR — Worker failed (exit {p.returncode})')
        sys.exit(p.returncode)

    print(f'[Main] Worker completed successfully.')

    elapsed = time.time() - t_start
    total_samples = len(sample_cre_ids)
    print(f'[Main] ALL COMPLETE in {elapsed:.0f}s ({elapsed / 60:.1f} min)')
    print(f'[Main] Output directory: {output_dir}')
    print(f'[Main]   embeddings.memmap    — shape {memmap_shape}, float32')
    print(f'[Main]   cre_info.npy          — {n_unique} unique CREs')
    print(f'[Main]   cre_to_emb_idx.pkl    — CRE id → row index')
    print(f'[Main]   sample_cre_ids.npy    — {total_samples} P0-filtered samples')
    print(f'[Main]   sample_split.npy      — split label per sample')
    print(f'[Main]   unique_cre_split.npy  — split label per unique CRE')
    print(f'[Main] Split: train={split_counts["train"]}, '
          f'valid={split_counts["valid"]}, test={split_counts["test"]}')
    print(f'[Main] Split ratios: '
          f'{split_counts["train"] / n_unique * 100:.1f}% / '
          f'{split_counts["valid"] / n_unique * 100:.1f}% / '
          f'{split_counts["test"] / n_unique * 100:.1f}%')


if __name__ == '__main__':
    main()
