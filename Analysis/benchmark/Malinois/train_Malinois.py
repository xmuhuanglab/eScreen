#!/usr/bin/env python3
"""
Malinois (BassetBranched) — 5-head model for 5 cell lines, 512bp input.

Usage:
    python train_Malinois.py --device cuda:0 --epochs 50 --batch_size 128
"""
import argparse, os, sys, gc, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from data_utils import (
    build_datasets_from_original, prepare_sequence_1d,
    per_cell_line_metrics, print_metrics_table, save_metrics,
    P0_CELL_LINES, CELL_LINE_TO_IDX
)


# ── Model ───────────────────────────────────────────────────────────
class BassetBranched5Head(nn.Module):
    """BassetBranched backbone with 5 separate output heads (one per cell line)."""

    def __init__(self, input_len=512):
        super().__init__()
        # Shared conv backbone
        self.conv_block = nn.Sequential(
            nn.Conv1d(4, 300, kernel_size=19, padding=9),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(300, 200, kernel_size=11, padding=5),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(200, 200, kernel_size=7, padding=3),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # Compute conv output length
        conv_output_len = input_len
        conv_output_len = conv_output_len // 3
        conv_output_len = conv_output_len // 4
        conv_output_len = conv_output_len // 4

        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * conv_output_len, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 5 cell-line-specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1000, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            ) for _ in range(5)
        ])

    def forward(self, x):
        # x: (B, 4, L)
        x = self.conv_block(x)
        x = self.shared_fc(x)
        logits = torch.cat([head(x) for head in self.heads], dim=1)  # (B, 5)
        return logits


# ── Training ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        x, labels, cl_idx = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(x)  # (B, 5)
        # Supervise only the head matching the cell line
        loss = loss_fn(outputs[torch.arange(len(cl_idx)), cl_idx], labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict(model, data_dict, batch_size, device):
    """Return predictions (logits) for the matching head of each sample."""
    model.eval()
    all_preds = []
    seq = prepare_sequence_1d(data_dict, seq_len=512).to(device)
    cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long, device=device)

    for i in range(0, len(seq), batch_size):
        x_batch = seq[i:i + batch_size]
        cl_batch = cl_idx[i:i + batch_size]
        outputs = model(x_batch)
        # Take the head matching each sample's cell line
        preds = outputs[torch.arange(len(cl_batch)), cl_batch]
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


# ── Data Loader ─────────────────────────────────────────────────────
class EScreenDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, seq_len=512):
        self.seq = prepare_sequence_1d(data_dict, seq_len=seq_len)
        self.labels = torch.tensor(data_dict['label'], dtype=torch.float)
        self.cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seq[idx], self.labels[idx], self.cl_idx[idx]


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Malinois — 5-head BassetBranched')
    parser.add_argument('--device', default='cuda:0', help='GPU device, e.g. cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--check_step', type=int, default=500,
                        help='Validate every N batches')
    parser.add_argument('--output_dir', default='./Models', help='Where to save model weights')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f'[Malinois] Device: {device}, Cell lines: {P0_CELL_LINES}')
    print(f'[Malinois] Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}')

    # 1. Data
    print('\n[1/4] Loading data...')
    trainset, validset, testset, locusset = build_datasets_from_original()

    train_loader = torch.utils.data.DataLoader(
        EScreenDataset(trainset, seq_len=512),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # 2. Model
    print('\n[2/4] Building model...')
    torch.cuda.empty_cache()
    gc.collect()
    model = BassetBranched5Head(input_len=512).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total_params:,}, Trainable: {trainable:,}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # 3. Train
    print('\n[3/4] Training...')
    save_name = f'{args.output_dir}/Malinois_5head'
    best_val_score = -float('inf')
    count = 0
    valid_count = 0
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            x, labels, cl_idx = [b.to(device) for b in batch]

            model.train()
            optimizer.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs[torch.arange(len(cl_idx)), cl_idx], labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            valid_count += 1

            avg_loss = epoch_loss / n_batches
            print(f"Epoch [{epoch+1}/{args.epochs}] Step {n_batches}, Loss: {avg_loss:.4f}", end='\r')

            # Validation check
            if valid_count >= args.check_step:
                pv = predict(model, validset, args.batch_size, device)
                results = per_cell_line_metrics(1 / (1 + np.exp(-pv)), validset)
                valid_results = [r for r in results if not np.isnan(r['AUROC'])]
                val_score = np.mean([r['AUROC'] for r in valid_results]) if valid_results else 0

                if val_score > best_val_score:
                    best_val_score = val_score
                    count = 0
                    torch.save(model.state_dict(), save_name + '.best.pt')
                    print(f"\n  Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val mean AUROC: {val_score:.4f}  ↑")
                else:
                    count += 1
                    print(f"\n  Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val mean AUROC: {val_score:.4f}  -")
                    if count >= args.earlystop:
                        print(f'  Early stop at epoch {epoch+1}, best val AUROC: {best_val_score:.4f}')
                        break
                valid_count = 0

        torch.save(model.state_dict(), f'{save_name}.epoch{epoch}.pt')
        if count >= args.earlystop:
            break

    # Load best model
    best_path = save_name + '.best.pt'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # 4. Evaluate
    print('\n[4/4] Evaluating on test set...')
    pt = predict(model, testset, args.batch_size, device)
    pt_sigmoid = 1 / (1 + np.exp(-pt))

    results = per_cell_line_metrics(pt_sigmoid, testset)
    print_metrics_table(results, 'Test Set')

    extra = f'Malinois (BassetBranched) 5-head\nSplit: original randomCRE.4.pkl → id-level 7:2:1 + P0 filter\n'
    extra += f'Cell lines: {P0_CELL_LINES}\nEpochs: {args.epochs}, LR: {args.lr}\n'
    save_metrics(results, save_name, extra)

    # Locus holdout
    if locusset is not None and locusset['length'] > 0:
        pl = predict(model, locusset, args.batch_size, device)
        pl_sigmoid = 1 / (1 + np.exp(-pl))
        locus_results = per_cell_line_metrics(pl_sigmoid, locusset)
        print_metrics_table(locus_results, 'Locus Holdout (chr11)')

    print('\nDone!')


if __name__ == '__main__':
    main()
