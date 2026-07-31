#!/usr/bin/env python3
"""
Train Enformer-embedding + cell-embedding + MLP classifier using
the NEW patch-based split dataset from extraEnformerEmb.py.

Adapted from train_enformer_emb.py for the new data format:
  - Embeddings: enformer_embeddings/embeddings.memmap
  - Sample splits: enformer_embeddings/sample_split.npy  (0=train, 1=valid, 2=test)
  - Sample CRE IDs: enformer_embeddings/sample_cre_ids.npy
  - No locus holdout set in the new split.
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
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def load_original_data():
    """Load the original 8-tuple pickle."""
    print(f'[Data] Loading original 8-tuple from {ORIGINAL_PATH}')
    with open(ORIGINAL_PATH, 'rb') as f:
        bundle = _NumpyCompatUnpickler(f).load()
    return bundle[:8]


# ══════════════════════════════════════════════════════════
#  Model (same as original train_enformer_emb.py)
# ══════════════════════════════════════════════════════════

class EnformerCellMLP(nn.Module):
    """Enformer embedding + cell embedding → MLP → binary prediction."""

    def __init__(self, enformer_dim=ENFORMER_DIM, cre_bins=CRE_BINS,
                 num_cell_lines=N_CELL_LINES, cell_emb_dim=CELL_EMB_DIM):
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
    cl = str(cl)
    return P0_RELABEL.get(cl, cl)


def _cell_idx(cl):
    return CELL_LINE_TO_IDX[_relabel(cl)]


def _is_p0(idx, cre_sub, p0_set):
    cl = str(cre_sub.loc[idx, 'cell_line'])
    if cl.upper() == 'WTC11':
        return 'hPSC' in P0_RELABEL.values()
    return cl in p0_set


def load_data(device):
    """Load data using the new patch-based split from extraEnformerEmb.py."""
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
