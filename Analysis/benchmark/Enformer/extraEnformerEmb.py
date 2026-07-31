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
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


# ══════════════════════════════════════════════════════════
#  Step 1 — Data Loading & P0 Filtering
# ══════════════════════════════════════════════════════════

def load_original_data():
    """Load the original 8-tuple pickle."""
    print(f'[Data] Loading original 8-tuple from {ORIGINAL_PATH}')
    with open(ORIGINAL_PATH, 'rb') as f:
        bundle = _NumpyCompatUnpickler(f).load()
    return bundle[:8]


def get_unique_cre_data():
    """
    Full pipeline: load → filter → P0 → dedup by CRE id.
    No id-level train/val/test split (replaced by patch-level split later).

    Returns:
      unique_cre      — DataFrame (id, chr, start, end) one row per unique CRE
      cre_to_emb      — dict  cre_id → embedding row index
      sample_cre_ids  — ndarray  CRE id of each P0-filtered sample
      p0_meta         — DataFrame of all P0-filtered samples
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

    Returns:
      cre_to_split  — dict  cre_id → 'train'|'valid'|'test'
      split_counts  — dict  split_name → int count
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
    """One-hot encode DNA string via uint8 lookup table."""
    alphabet = 'ACGT'
    neutral = 'N'

    def _to_uint8(s):
        return np.frombuffer(s.encode('ascii'), dtype=np.uint8)

    table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    table[_to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    table[_to_uint8(neutral)] = 0.0
    return table[_to_uint8(sequence.upper())]


def extract_sequence(fasta, chrom, start_0based, end_0based):
    """
    Extract DNA sequence from reference genome (pyfaidx).
    Pads with N at chromosome boundaries.
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
