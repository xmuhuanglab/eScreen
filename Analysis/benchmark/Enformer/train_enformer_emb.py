#!/usr/bin/env python3
"""
Train a simple Enformer-embedding + cell-embedding + MLP classifier.
Uses precomputed Enformer trunk embeddings (CRE bins 446:450).

Usage:
    python train_enformer_emb.py --device cuda:0

What this script implements
---------------------------
The decoder half of the "CRE-centered embedding finetune" strategy in
Supplementary Methods §6.2.1: Enformer is frozen and only a small head is
trained on its truncated trunk embedding. Enformer itself is never loaded here —
the features come from the ``embeddings.memmap`` written by
``extraEnformerEmb.py`` (the upstream Enformer SavedModel is required only for
that extraction step, not for this script).

Per CRE the head receives a ``(4, 3072)`` slice of Enformer's trunk output — four
128 bp bins, i.e. the 512 bp footprint the other benchmarked models see — flattens
it to 12,288 values, concatenates a learned cell-line embedding, and decodes to a
single logit. This mirrors the manuscript's "cell line encodings were generated
using torch's embedding layer, concatenated, and then decoded through an MLP".

Where the code and the manuscript disagree
------------------------------------------
* The docstring line above ("CRE bins 446:450") describes the *earlier* per-CRE
  extraction, where a fixed bin window was cut out of a 896-bin output. The
  current extraction is patch-based and stores four left-aligned bins near the
  CRE (see ``extraEnformerEmb.py``); bins 446:450 are not a meaningful constant
  any more.
* §6.2.1 quotes a 131,072 bp CRE-centred input; the patch-based extraction uses
  Enformer's native 393,216 bp field. See ``extraEnformerEmb.py``.

Expected on disk
----------------
All paths are relative to the working directory and must be produced by
``export_p0_samples.py`` + ``extraEnformerEmb.py`` first::

    ./enformer_embeddings/p0_filtered_samples.pkl   # exported sample table
    ./enformer_embeddings/embeddings.memmap         # (n_unique, 4, 3072) f32
    ./enformer_embeddings/cre_to_emb_idx.pkl        # cre_id -> memmap row

Outputs: ``./enformer_embeddings/enformer_emb_mlp_test_results.csv`` (via
``save_metrics``); model weights are **not** saved.

External dependency not vendored in this repository
---------------------------------------------------
``data_utils`` is imported from ``../_shared/``, a directory that does **not**
exist in this repository snapshot. The import provides
``per_cell_line_metrics``, ``print_metrics_table``, ``save_metrics``,
``P0_CELL_LINES`` and ``CELL_LINE_TO_IDX``; the script cannot run without that
module. Note that ``per_cell_line_metrics`` here is *not* the
``alphagenome_common`` function of the same name — it takes
``(probs, data_dict)`` in two arguments, as the call sites show.
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
EMBEDDINGS_DIR = './enformer_embeddings'
SAMPLE_CSV = f'{EMBEDDINGS_DIR}/p0_filtered_samples.pkl'
MEMMAP_PATH = f'{EMBEDDINGS_DIR}/embeddings.memmap'
CRE_TO_EMB_PATH = f'{EMBEDDINGS_DIR}/cre_to_emb_idx.pkl'

# ── Constants (same as data_utils) ──
P0_RELABEL = {'WTC11': 'hPSC'}
N_CELL_LINES = len(P0_CELL_LINES)  # 5
CRE_BINS = 4
ENFORMER_DIM = 3072
CELL_EMB_DIM = 32


# ══════════════════════════════════════════════════════════
#  Model
# ══════════════════════════════════════════════════════════

class EnformerCellMLP(nn.Module):
    """Enformer embedding + cell embedding → MLP → binary prediction.

    Parameters
    ----------
    enformer_dim : int, optional
        Width of one Enformer trunk bin (3072 for the released human model).
    cre_bins : int, optional
        Number of bins taken per CRE (4, i.e. 512 bp at 128 bp/bin).
    num_cell_lines : int, optional
        Number of distinct cell lines (5: A375, HT29, HepG2, K562, hPSC).
    cell_emb_dim : int, optional
        Size of the learned cell-line embedding.

    Attributes
    ----------
    enformer_proj : torch.nn.Sequential
        ``Linear(3072 * 4 -> 256)`` + ReLU, the only path that sees sequence
        information. Its 12,288 inputs are frozen features, so this layer carries
        essentially all of the model's capacity.
    cell_embedding : torch.nn.Embedding
        Cell-line lookup table, ``(5, 32)``.
    mlp : torch.nn.Sequential
        ``256 + 32 -> 128 -> 64 -> 1`` with one dropout (p=0.15) after the first
        block.

    Notes
    -----
    * The head is intentionally tiny (< 3.2 M parameters, dominated by the
      3072 -> 256 projection). With frozen upstream features, extra depth would
      mostly overfit the ~200 k training rows.
    * The forward returns **logits**; every call site applies ``sigmoid`` itself
      and trains with ``binary_cross_entropy_with_logits`` (numerically stabler
      than a sigmoid-then-BCE formulation).
    * ``reshape(B, -1)`` flattens ``(4, 3072)`` in bin-major order, so bin order
      matters and must match between extraction and training.
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
        ``ReLU(inplace=True)`` is safe here because no tensor is reused after the
        activation, and it saves a temporary per layer.
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
        """
        Args:
            enformer_emb: (B, 4, 3072)
            cell_line_idx: (B,) int
        Returns:
            logits: (B,)

        Parameters
        ----------
        enformer_emb : torch.Tensor
            Frozen trunk embeddings, shape ``(B, 4, 3072)``, ``float32``.
        cell_line_idx : torch.Tensor
            Cell-line indices into :attr:`cell_embedding`, shape ``(B,)``,
            ``int64``.

        Returns
        -------
        torch.Tensor
            Logits, shape ``(B,)`` (the trailing singleton is squeezed away).

        Notes
        -----
        No activation on the output: the loss function is
        ``binary_cross_entropy_with_logits``, and ``evaluate`` applies ``sigmoid``
        exactly once before computing metrics.
        """
        B = enformer_emb.shape[0]
        emb_flat = enformer_emb.reshape(B, -1)                 # (B, 12288)
        emb_proj = self.enformer_proj(emb_flat)                # (B, 256)
        cell_emb = self.cell_embedding(cell_line_idx)          # (B, 32)
        combined = torch.cat([emb_proj, cell_emb], dim=1)      # (B, 288)
        logits = self.mlp(combined).squeeze(-1)                # (B,)
        return logits


# ══════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════

def _relabel(cl):
    """Map a raw cell-line label to its benchmark name (``WTC11`` -> ``hPSC``).

    Parameters
    ----------
    cl : str
        Cell-line value as stored in the sample table.

    Returns
    -------
    str
        The canonical label; unchanged if not in :data:`P0_RELABEL`.

    Notes
    -----
    ``str(cl)`` guards against NaN/None arriving from a partially joined table;
    such a value would then fail in :func:`_cell_idx` rather than deep inside the
    embedding lookup.
    """
    cl = str(cl)
    return P0_RELABEL.get(cl, cl)


def _cell_idx(cl):
    """Convert a cell-line label to its integer embedding index.

    Parameters
    ----------
    cl : str
        Raw cell-line label.

    Returns
    -------
    int
        Index into ``CELL_LINE_TO_IDX`` (and therefore into the model's
        ``nn.Embedding``), in ``P0_CELL_LINES`` order.

    Notes
    -----
    KeyError here means the label was neither a P0 cell line nor relabelled —
    worth failing loudly, since silently mapping it to index 0 would train the
    wrong head.
    """
    return CELL_LINE_TO_IDX[_relabel(cl)]


def load_data(device):
    """Load all data, build per-split tensors, return dict of splits.

    Parameters
    ----------
    device : torch.device
        Unused (tensors are built on CPU and moved per batch); kept for call-site
        symmetry.

    Returns
    -------
    dict
        ``{split_name: {'emb', 'cl_idx', 'label', 'n'} | None}`` for
        ``'train'``, ``'valid'``, ``'test'`` and ``'locus'``. ``emb`` is
        ``(n, 4, 3072)`` float32, ``cl_idx`` is ``(n,)`` int64, ``label`` is
        ``(n,)`` float32. A split with no rows maps to ``None``.

    Notes
    -----
    * The whole embedding matrix is copied out of the memmap into RAM once
      (``np.array(emb_memmap)``) and then indexed by the per-split embedding
      indices. For ~122 k CREs that is ~5.9 GB — the reason the extraction script
      used a memmap in the first place.
    * Row selection is *gathered* (`emb_full[emb_indices]`), so the resulting
      tensors duplicate memory: embeddings are shared across cell lines in the
      memmap but materialised once per sample here.
    * Splits come from the exported table's ``split`` column, i.e. the
      identifier-level 7:2:1 split of ``export_p0_samples.py`` plus the ``locus``
      holdout — not from the patch split.
    * The split names are matched exactly, so ``'train'``/``'valid'``/``'test'``/
      ``'locus'`` must be spelled as in the CSV.
    """
    print('[Data] Loading sample metadata...')
    samples = pd.read_pickle(SAMPLE_CSV)

    print('[Data] Loading CRE → embedding index mapping...')
    with open(CRE_TO_EMB_PATH, 'rb') as f:
        cre_to_emb = pickle.load(f)

    print('[Data] Loading Enformer embeddings from memmap...')
    n_unique = len(cre_to_emb)
    emb_memmap = np.memmap(MEMMAP_PATH, dtype=np.float32, mode='r',
                           shape=(n_unique, CRE_BINS, ENFORMER_DIM))
    emb_full = torch.from_numpy(np.array(emb_memmap))  # (121729, 4, 3072) in CPU RAM
    print(f'  Embedding tensor shape: {emb_full.shape}, '
          f'RAM: {emb_full.numel() * 4 / 1e9:.2f} GB')

    # Build per-split data
    splits = {}
    for split_name in ['train', 'valid', 'test', 'locus']:
        mask = samples['split'] == split_name
        sub = samples[mask]
        n = len(sub)
        if n == 0:
            splits[split_name] = None
            continue

        # Embedding indices
        emb_indices = torch.tensor(
            [cre_to_emb[cid] for cid in sub['id'].values], dtype=torch.long
        )
        # Embeddings
        embs = emb_full[emb_indices]  # (n, 4, 3072)

        # Cell line indices
        cl_idx = torch.tensor(
            [_cell_idx(cl) for cl in sub['cell_line'].values], dtype=torch.long
        )

        # Labels
        labels = torch.tensor(sub['label'].values, dtype=torch.float32)

        splits[split_name] = {
            'emb': embs,          # (n, 4, 3072)
            'cl_idx': cl_idx,     # (n,)
            'label': labels,      # (n,)
            'n': n,
        }
        pos_rate = labels.mean().item()
        print(f'  {split_name}: n={n}, pos_rate={pos_rate:.4f}')

    return splits


# ══════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════

def train_epoch(model, data, optimizer, device):
    """Run one full pass over the training split.

    Parameters
    ----------
    model : EnformerCellMLP
        Model in training mode.
    data : dict
        The ``'train'`` entry of :func:`load_data`'s return value.
    optimizer : torch.optim.Optimizer
        AdamW, constructed in :func:`run`.
    device : torch.device
        Where the tensors are moved for the pass.

    Returns
    -------
    float
        Mean batch loss over the epoch (batch-size weighted implicitly: every
        batch contributes equally regardless of size, so the final short batch
        counts as much as a full one).

    Notes
    -----
    * The split is **fully resident on the GPU** for the pass
      (``data['emb'].to(device)``); this is only viable because the head consumes
      precomputed features and the embedding tensor fits in GPU memory.
    * ``torch.randperm(n, device=device)`` generates the permutation on the GPU,
      avoiding a host-to-device copy per epoch.
    * Standard BCE-with-logits, no class weighting and no sampler: the
      oversampling/label-boosting machinery of Supplementary Methods §2.3.4 is
      used for the eScreen model, not for this baseline head.
    * If ``n`` is 0 (empty split) the loop body never runs and the division by
      ``n_batches == 0`` raises; callers are expected to have checked ``n``.
    """
    model.train()
    emb = data['emb'].to(device)
    cl_idx = data['cl_idx'].to(device)
    labels = data['label'].to(device)

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
        Model; switched to ``eval()`` for the duration.
    data : dict
        One split from :func:`load_data`.
    device : torch.device
        Device to evaluate on.

    Returns
    -------
    loss : float
        Mean BCE-with-logits over the whole split (a single reduction over all
        rows, not the mean of batch means).
    probs : numpy.ndarray
        ``sigmoid(logits)``, shape ``(n,)``, float32.
    labels : numpy.ndarray
        Ground-truth labels, shape ``(n,)``.
    cl_idx : numpy.ndarray
        Cell-line indices, shape ``(n,)``; callers map them back through
        ``P0_CELL_LINES`` before computing per-cell-line metrics.

    Notes
    -----
    ``@torch.no_grad()`` disables autograd for the whole function. Batching is
    sequential slicing (``slice(i, i+512)``) rather than an index tensor because
    no shuffling is needed and slicing is cheaper.

    ``sigmoid`` is applied here, once, so every downstream metric function
    receives probabilities while training still operates on logits.
    """
    model.eval()
    emb = data['emb'].to(device)
    cl_idx = data['cl_idx'].to(device)
    labels = data['label'].to(device)

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
    """Parse CLI arguments, train the head, and report test / locus metrics.

    Returns
    -------
    None

    Notes
    -----
    * Model selection uses mean **validation AUROC** averaged across cell lines,
      with ``ReduceLROnPlateau(mode='max')`` halving the learning rate after 5
      stagnant epochs and early stopping after 10 (``--early_stop``). The
      manuscript's m2a composite (§5.5) is *not* used for this baseline.
    * The mean skips ``nan`` AUROCs (cell lines with a single class in
      validation) — without the filter the mean would be ``nan`` and no epoch
      would ever improve.
    * The best state dict is kept in RAM (``best_state``), so nothing is written
      to disk except the metrics CSV. A crash loses the trained weights.
    * ``model.load_state_dict(best_state)`` happens *after* the loop; if the loop
      never improved (``best_state`` stays ``None``) this raises.
    * The locus holdout is evaluated only for reporting; it is never used for
      selection.
    * The last-epoch counter ``epoch`` is written into the metrics header, so the
      recorded "Epochs" is the number actually run, not necessarily the best one.
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
    locus_data = splits['locus']

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

        # Per-cell-line AUROC on validation set
        valid_dict = {'label': val_labels, 'cell_line': val_cl}
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
    print('Test Set Evaluation')
    print(f'{"=" * 60}')
    test_loss, test_probs, test_labels, test_cl = evaluate(model, test_data, device)
    test_dict = {'label': test_labels, 'cell_line': [P0_CELL_LINES[i] for i in test_cl]}
    test_results = per_cell_line_metrics(test_probs, test_dict)
    print_metrics_table(test_results, 'Test Set')

    # ── Evaluate on locus holdout ──
    if locus_data is not None and locus_data['n'] > 0:
        print(f'\n{"=" * 60}')
        print('Locus Holdout Evaluation')
        print(f'{"=" * 60}')
        locus_loss, locus_probs, locus_labels, locus_cl = evaluate(
            model, locus_data, device)
        locus_dict = {'label': locus_labels, 'cell_line': [P0_CELL_LINES[i] for i in locus_cl]}
        locus_results = per_cell_line_metrics(locus_probs, locus_dict)
        print_metrics_table(locus_results, 'Locus Holdout')

    # ── Save results ──
    save_name = f'{EMBEDDINGS_DIR}/enformer_emb_mlp'
    extra_info = (f'Model: EnformerEmb+CellEmb+MLP\n'
                  f'Epochs: {epoch}\nBest val AUROC: {best_val_score:.4f}\n'
                  f'Cell lines: {P0_CELL_LINES}\n')
    save_metrics(test_results, save_name, extra_info=extra_info)

    print(f'\nAll done. Results saved to {save_name}_test_results.csv')


if __name__ == '__main__':
    run()
