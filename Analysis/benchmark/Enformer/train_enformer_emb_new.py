#!/usr/bin/env python3
"""
Train Enformer-embedding + cell-embedding + MLP classifier using
the NEW patch-based split dataset from extraEnformerEmb.py.

Adapted from train_enformer_emb.py for the new data format:
  - Embeddings: enformer_embeddings/embeddings.memmap
  - Sample splits: enformer_embeddings/sample_split.npy  (0=train, 1=valid, 2=test)
  - Sample CRE IDs: enformer_embeddings/sample_cre_ids.npy
  - No locus holdout set in the new split.

What this script implements
---------------------------
The same decoder as ``train_enformer_emb.py`` (Supplementary Methods §6.2.1,
embedding fine-tuning with a frozen Enformer), but wired to the patch-based
extraction of ``extraEnformerEmb.py``: it reads the split labels that script
wrote instead of the identifier-level split stored in the exported CSV.

Because the split is spatial (CREs sharing a 114,688 bp patch always share a
split), this variant is the leakage-safe one for long-range regulatory context,
but it is *not* the split used elsewhere in the benchmark — see the warning in
``extraEnformerEmb.py``. Results are therefore not directly comparable with the
other models' numbers.

Expected on disk
----------------
Inputs::

    # Hard-coded absolute path on the authors' cluster (adjust before rerunning);
    # only the (cell_line, label) columns and the row index are needed.
    /cluster/huanglab/sluo/sluoo/escreen/eScreen_training/datasets/
        TestData_20260509.randomCRE.4.pkl

    ./enformer_embeddings/embeddings.memmap        # (n_unique, 4, 3072) f32
    ./enformer_embeddings/cre_to_emb_idx.pkl        # cre_id -> memmap row
    ./enformer_embeddings/sample_cre_ids.npy        # CRE id per sample
    ./enformer_embeddings/sample_split.npy          # 0=train, 1=valid, 2=test

Outputs: ``./enformer_embeddings/enformer_emb_mlp_patch_split_test_results.csv``.
No weights are saved.

External dependency not vendored in this repository
---------------------------------------------------
``data_utils`` from ``../_shared/`` — that directory does not exist in this
repository snapshot (``per_cell_line_metrics``, ``print_metrics_table``,
``save_metrics``, ``P0_CELL_LINES``, ``CELL_LINE_TO_IDX``).
"""

import argparse, os, sys, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from data_utils import (
    per_cell_line_metrics, print_metrics_table, save_metrics,
    P0_CELL_LINES, CELL_LINE_TO_IDX,
)

# ── Paths ──
ORIGINAL_PATH = "/cluster/huanglab/sluo/sluoo/escreen/eScreen_training/datasets/TestData_20260509.randomCRE.4.pkl"
EMBEDDINGS_DIR = './enformer_embeddings'
MEMMAP_PATH = f'{EMBEDDINGS_DIR}/embeddings.memmap'
CRE_TO_EMB_PATH = f'{EMBEDDINGS_DIR}/cre_to_emb_idx.pkl'
SAMPLE_CRE_IDS_PATH = f'{EMBEDDINGS_DIR}/sample_cre_ids.npy'
SAMPLE_SPLIT_PATH = f'{EMBEDDINGS_DIR}/sample_split.npy'

# ── Constants (same as data_utils) ──
P0_RELABEL = {'WTC11': 'hPSC'}
N_CELL_LINES = len(P0_CELL_LINES)  # 5
CRE_BINS = 4
ENFORMER_DIM = 3072
CELL_EMB_DIM = 32
RRA_BOTTOM = 0.5
MIN_CELL_LINES = 2


# ── Numpy compat (old pickle format) ──
class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that rewrites ``numpy._core`` module paths to ``numpy.core``.

    Allows the pre-numpy-2.0 eScreen bundle to be loaded under numpy 2.x (or the
    reverse) without pinning a numpy version; ``numpy.core`` remains a
    compatibility alias.

    Parameters
    ----------
    Inherits ``pickle.Unpickler``'s constructor (a file-like object).
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
        Only ``numpy._core`` (numpy >= 2.0) is rewritten; other modules are
        delegated to the base implementation unchanged.
        """
        if module.startswith('numpy._core'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def load_original_data():
    """Load the original 8-tuple pickle.

    Returns
    -------
    tuple
        ``(trainset, validset, testset, train_idx, valid_idx, test_idx,
        cre_sub, final_df)`` — the first eight elements of the bundle. This script
        uses only ``*_idx`` and ``cre_sub``.

    Notes
    -----
    See the note in ``extraEnformerEmb.py``: the pickle was written by a
    different numpy major version, hence :class:`_NumpyCompatUnpickler`.
    """
    print(f'[Data] Loading original 8-tuple from {ORIGINAL_PATH}')
    with open(ORIGINAL_PATH, 'rb') as f:
        bundle = _NumpyCompatUnpickler(f).load()
    return bundle[:8]


# ══════════════════════════════════════════════════════════
#  Model (same as original train_enformer_emb.py)
# ══════════════════════════════════════════════════════════

class EnformerCellMLP(nn.Module):
    """Enformer embedding + cell embedding → MLP → binary prediction.

    Byte-for-byte the same architecture as ``train_enformer_emb.py``'s class (the
    two files are intentionally independent so either can be run alone). See that
    docstring for the layer-by-layer description; the forward pass returns logits
    of shape ``(B,)`` from a flattened ``(B, 4, 3072)`` embedding plus a
    ``(B,)`` cell-line index.

    Notes
    -----
    The uniform-looking attribute names (``enformer_proj``, ``cell_embedding``,
    ``mlp``) matter: checkpoints saved from one script's model would load into
    the other's.
    """

    def __init__(self, enformer_dim=ENFORMER_DIM, cre_bins=CRE_BINS,
                 num_cell_lines=N_CELL_LINES, cell_emb_dim=CELL_EMB_DIM):
        """Build the projection, the cell-line table and the MLP head.

        Parameters
        ----------
        enformer_dim : int, optional
            Width of one Enformer trunk bin (3072).
        cre_bins : int, optional
            Bins taken per CRE (4); the projection's input size is
            ``enformer_dim * cre_bins``.
        num_cell_lines : int, optional
            Rows in the cell-line embedding table (5).
        cell_emb_dim : int, optional
            Width of that embedding (32).

        Notes
        -----
        Layer shapes and names must stay identical to
        ``train_enformer_emb.EnformerCellMLP`` so a checkpoint trained by either
        script loads into the other.
        """
        super().__init__()
        self.enformer_proj = nn.Sequential(
            nn.Linear(enformer_dim * cre_bins, 256),
            nn.ReLU(inplace=True),
        )
        self.cell_embedding = nn.Embedding(num_cell_lines, cell_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(256 + cell_emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, enformer_emb, cell_line_idx):
        """Compute per-sample logits.

        Parameters
        ----------
        enformer_emb : torch.Tensor
            ``(B, 4, 3072)`` frozen trunk embeddings.
        cell_line_idx : torch.Tensor
            ``(B,)`` int64 indices into the cell-line embedding table.

        Returns
        -------
        torch.Tensor
            ``(B,)`` logits (no sigmoid).
        """
        B = enformer_emb.shape[0]
        emb_flat = enformer_emb.reshape(B, -1)
        emb_proj = self.enformer_proj(emb_flat)
        cell_emb = self.cell_embedding(cell_line_idx)
        combined = torch.cat([emb_proj, cell_emb], dim=1)
        logits = self.mlp(combined).squeeze(-1)
        return logits


# ══════════════════════════════════════════════════════════
#  Data Loading — adapted for new patch-based split
# ══════════════════════════════════════════════════════════

def _relabel(cl):
    """Map a raw cell-line label to its benchmark name (``WTC11`` -> ``hPSC``).

    Parameters
    ----------
    cl : str
        Raw cell-line value from the table.

    Returns
    -------
    str
        Canonical label, unchanged when not in :data:`P0_RELABEL`.
    """
    cl = str(cl)
    return P0_RELABEL.get(cl, cl)


def _cell_idx(cl):
    """Map a raw cell-line label to its integer embedding index.

    Parameters
    ----------
    cl : str
        Raw cell-line value.

    Returns
    -------
    int
        Index in ``P0_CELL_LINES`` order (via ``CELL_LINE_TO_IDX``).
    """
    return CELL_LINE_TO_IDX[_relabel(cl)]


def _is_p0(idx, cre_sub, p0_set):
    """Return True if a CRE-table row belongs to one of the five P0 cell lines.

    Parameters
    ----------
    idx : int
        *Label* index into ``cre_sub``.
    cre_sub : pandas.DataFrame
        The full CRE table from the original bundle.
    p0_set : set of str
        The five accepted cell-line labels.

    Returns
    -------
    bool
        True for a P0 cell line, and for WTC11 (relabelled to hPSC).

    Notes
    -----
    This must reproduce ``extraEnformerEmb._is_p0`` exactly: the assertion in
    :func:`load_data` checks only the *length* of the reconstructed sample list,
    so a filter that differs subtly (same count, different rows) would pass and
    silently pair embeddings with the wrong metadata.
    """
    cl = str(cre_sub.loc[idx, 'cell_line'])
    if cl.upper() == 'WTC11':
        return 'hPSC' in P0_RELABEL.values()
    return cl in p0_set


def load_data(device):
    """Load data using the new patch-based split from extraEnformerEmb.py.

    Parameters
    ----------
    device : torch.device
        Unused (tensors are built on CPU); kept for symmetry with the caller.

    Returns
    -------
    dict
        ``{'train'|'valid'|'test': {'emb', 'cl_idx', 'label', 'n'}}``. A split
        with no samples maps to ``None``.

    Notes
    -----
    * Two separate sources have to be reconciled: **metadata** (cell line, label,
      interval) is rebuilt from the original pickle, while **splits and
      embeddings** come from ``extraEnformerEmb.py``'s ``.npy``/memmap files.
      They are stitched together positionally — ``sample_cre_ids`` and
      ``sample_split`` are assumed to be in the same order as the reconstructed
      P0-filtered list.
    * The two ``assert`` statements only compare *lengths*. They would not catch a
      reordering, which is why ``_is_p0`` must match the extraction script
      exactly. This is the most fragile point of the script.
    * ``p0_meta = cre_sub.loc[p0_filtered]`` keeps the original index, so
      ``p0_meta[mask]`` (boolean array from ``sample_split``) relies on the index
      being ordered like ``p0_filtered`` — true because ``.loc`` preserves order.
    * ``cre_to_emb.get(cid, 0)`` silently falls back to embedding row 0 for an
      unknown CRE id: a missing id corrupts that sample's features instead of
      raising. (The ``0`` row is a real CRE, so this is a quiet data bug, not an
      obvious blank.)
    * ``sample_split`` is matched against the integer codes 0/1/2 written by the
      extraction script; ``split_map`` mirrors ``SPLIT_LABELS`` there.
    """
    # ── 1. Load original cre_sub for cell_line/label mapping ──
    _, _, _, train_idx, valid_idx, test_idx, cre_sub, _ = load_original_data()
    pool_index = list(train_idx) + list(valid_idx) + list(test_idx)

    # ── 2. Reconstruct P0-filtered sample list (same logic as extraEnformerEmb.py) ──
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
    keep_indices = meta.loc[keep_mask, '_pool_idx'].astype(int).tolist()

    p0_set = set(P0_CELL_LINES)
    p0_filtered = [pool_index[i] for i in keep_indices
                   if _is_p0(pool_index[i], cre_sub, p0_set)]

    n_samples = len(p0_filtered)
    print(f'[Data] Reconstructed P0-filtered samples: {n_samples}')

    # ── 3. Load new splits & embeddings ──
    sample_cre_ids = np.load(SAMPLE_CRE_IDS_PATH, allow_pickle=True)
    sample_split = np.load(SAMPLE_SPLIT_PATH, allow_pickle=True)

    assert len(sample_cre_ids) == n_samples, \
        f'Mismatch: sample_cre_ids ({len(sample_cre_ids)}) vs reconstructed ({n_samples})'
    assert len(sample_split) == n_samples

    print('[Data] Loading CRE to embedding index mapping...')
    with open(CRE_TO_EMB_PATH, 'rb') as f:
        cre_to_emb = pickle.load(f)

    print('[Data] Loading Enformer embeddings from memmap...')
    n_unique = len(cre_to_emb)
    emb_memmap = np.memmap(MEMMAP_PATH, dtype=np.float32, mode='r',
                           shape=(n_unique, CRE_BINS, ENFORMER_DIM))
    emb_full = torch.from_numpy(np.array(emb_memmap))
    print(f'  Embedding tensor shape: {emb_full.shape}, '
          f'RAM: {emb_full.numel() * 4 / 1e9:.2f} GB')

    # ── 4. Get metadata for each sample ──
    p0_meta = cre_sub.loc[p0_filtered]

    # ── 5. Build per-split datasets ──
    splits = {}
    split_map = {'train': 0, 'valid': 1, 'test': 2}

    for split_name, split_val in split_map.items():
        mask = sample_split == split_val
        n = mask.sum()
        if n == 0:
            splits[split_name] = None
            continue

        sub_meta = p0_meta[mask]
        sub_ids = sample_cre_ids[mask]

        emb_indices = torch.tensor(
            [cre_to_emb.get(cid, 0) for cid in sub_ids], dtype=torch.long
        )
        embs = emb_full[emb_indices]

        cl_idx = torch.tensor(
            [_cell_idx(cl) for cl in sub_meta['cell_line'].values], dtype=torch.long
        )
        labels = torch.tensor(sub_meta['label'].values, dtype=torch.float32)

        splits[split_name] = {
            'emb': embs,
            'cl_idx': cl_idx,
            'label': labels,
            'n': n,
        }
        pos_rate = labels.mean().item()
        print(f'  {split_name}: n={n}, pos_rate={pos_rate:.4f}')

    return splits


# ══════════════════════════════════════════════════════════
#  Training & Evaluation
# ══════════════════════════════════════════════════════════

def train_epoch(model, data, optimizer, device):
    """Run one full pass over the training split.

    Parameters
    ----------
    model : EnformerCellMLP
        Model, set to train mode here.
    data : dict
        The ``'train'`` entry from :func:`load_data`.
    optimizer : torch.optim.Optimizer
        AdamW built in :func:`run`.
    device : torch.device
        Target device.

    Returns
    -------
    float
        Mean batch loss over the epoch (batches weighted equally, not by size).

    Notes
    -----
    Identical to the ``train_enformer_emb.py`` version except for
    ``non_blocking=True`` on the host-to-device copies; that only pays off with
    pinned memory, which these tensors do not use, so it is effectively a no-op
    here.
    """
    model.train()
    emb = data['emb'].to(device, non_blocking=True)
    cl_idx = data['cl_idx'].to(device, non_blocking=True)
    labels = data['label'].to(device, non_blocking=True)

    n = data['n']
    batch_size = 256
    perm = torch.randperm(n, device=device)
    total_loss = 0
    n_batches = 0

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        logits = model(emb[idx], cl_idx[idx])
        loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, data, device):
    """Evaluate the model on one split.

    Parameters
    ----------
    model : EnformerCellMLP
        Model to evaluate.
    data : dict
        One split from :func:`load_data`; must not be ``None``.
    device : torch.device
        Device to evaluate on.

    Returns
    -------
    loss : float
        BCE-with-logits reduced over all rows.
    probs : numpy.ndarray
        ``(n,)`` sigmoid probabilities.
    labels : numpy.ndarray
        ``(n,)`` ground-truth labels.
    cl_idx : numpy.ndarray
        ``(n,)`` cell-line indices, mapped back through ``P0_CELL_LINES`` by the
        caller.

    Notes
    -----
    Same reduction and batch size as the original script, so the two variants'
    validation losses are comparable.
    """
    model.eval()
    emb = data['emb'].to(device, non_blocking=True)
    cl_idx = data['cl_idx'].to(device, non_blocking=True)
    labels = data['label'].to(device, non_blocking=True)

    n = data['n']
    batch_size = 512
    all_logits = []

    for i in range(0, n, batch_size):
        idx = slice(i, min(i + batch_size, n))
        logits = model(emb[idx], cl_idx[idx])
        all_logits.append(logits)

    logits = torch.cat(all_logits)
    loss = F.binary_cross_entropy_with_logits(logits, labels).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy()
    cl_np = cl_idx.cpu().numpy()
    return loss, probs, labels_np, cl_np


def run():
    """Parse CLI arguments, train, and report test metrics for the patch split.

    Returns
    -------
    None

    Notes
    -----
    * Identical optimisation setup to ``train_enformer_emb.py`` (AdamW 1e-3,
      weight decay 1e-5, batch 256, ReduceLROnPlateau on max validation AUROC,
      early stop 10) so that any difference in the reported numbers is
      attributable to the split, not to the training recipe.
    * There is no locus split in the patch-based scheme, hence no locus block
      here; ``extraEnformerEmb.py`` assigns those CREs to train/valid/test like
      any other.
    * ``np.isnan`` filtering of per-cell-line AUROCs is required for the same
      reason as in the original script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--early_stop', type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f'Device: {device}')

    # ── Load data ──
    splits = load_data(device)
    train_data = splits['train']
    valid_data = splits['valid']
    test_data = splits['test']

    # ── Build model ──
    model = EnformerCellMLP().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[Model] Parameters: {n_params:,}')

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )

    # ── Training loop ──
    print(f'\n{"=" * 60}')
    print(f'Training: {args.epochs} epochs max, early stop {args.early_stop}')
    print(f'{"=" * 60}')

    best_val_score = -float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_data, optimizer, device)
        val_loss, val_probs, val_labels, val_cl = evaluate(model, valid_data, device)

        valid_dict = {'label': val_labels, 'cell_line': [P0_CELL_LINES[i] for i in val_cl]}
        results = per_cell_line_metrics(val_probs, valid_dict)
        val_auroc = np.mean([r['AUROC'] for r in results if not np.isnan(r['AUROC'])])

        scheduler.step(val_auroc)

        arrow = ' ↑' if val_auroc > best_val_score else ''
        if val_auroc > best_val_score:
            best_val_score = val_auroc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val mean AUROC: {val_auroc:.4f}{arrow}')

        if patience_counter >= args.early_stop:
            print(f'  Early stopping at epoch {epoch}')
            break

    # ── Restore best model ──
    model.load_state_dict(best_state)

    # ── Evaluate on test set ──
    print(f'\n{"=" * 60}')
    print('Test Set Evaluation  (Patch-based Split)')
    print(f'{"=" * 60}')
    test_loss, test_probs, test_labels, test_cl = evaluate(model, test_data, device)
    test_dict = {'label': test_labels, 'cell_line': [P0_CELL_LINES[i] for i in test_cl]}
    test_results = per_cell_line_metrics(test_probs, test_dict)
    print_metrics_table(test_results, 'Test Set (Patch-based Split)')

    # ── Save results ──
    save_name = f'{EMBEDDINGS_DIR}/enformer_emb_mlp_patch_split'
    extra_info = (f'Model: EnformerEmb+CellEmb+MLP (patch-based split)\n'
                  f'Epochs: {epoch}\nBest val AUROC: {best_val_score:.4f}\n'
                  f'Cell lines: {P0_CELL_LINES}\n')
    save_metrics(test_results, save_name, extra_info=extra_info)

    print(f'\nAll done. Results saved to {save_name}_test_results.csv')


if __name__ == '__main__':
    run()
