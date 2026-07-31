#!/usr/bin/env python3
"""
Train a simple Enformer-embedding + cell-embedding + MLP classifier.
Uses precomputed Enformer trunk embeddings (CRE bins 446:450).

Usage:
    python train_enformer_emb.py --device cuda:0
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
        """
        Args:
            enformer_emb: (B, 4, 3072)
            cell_line_idx: (B,) int
        Returns:
            logits: (B,)
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
    cl = str(cl)
    return P0_RELABEL.get(cl, cl)


def _cell_idx(cl):
    return CELL_LINE_TO_IDX[_relabel(cl)]


def load_data(device):
    """Load all data, build per-split tensors, return dict of splits."""
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
