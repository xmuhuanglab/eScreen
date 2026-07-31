#!/usr/bin/env python3
"""
Sei — 5-head model for 5 cell lines, 4096bp input.

The Sei architecture uses BSplineConv1D + dilated convolutions.
Input is center-padded from 512bp to 4096bp by default, or can be loaded
from a pre-existing 4096bp data file via --data_path.

Usage:
    python train_Sei.py --device cuda:0 --epochs 50 --batch_size 32
    python train_Sei.py --device cuda:1 --data_path /path/to/4096bp_data.pkl
"""
import argparse, os, sys, gc, pickle, warnings
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
    P0_CELL_LINES, CELL_LINE_TO_IDX, load_original_data
)

# Import Sei building blocks
from Sei import BSplineTransformation


# ── Model ───────────────────────────────────────────────────────────
class Sei5Head(nn.Module):
    """Sei backbone with 5 separate output heads, input length 4096bp."""

    def __init__(self, sequence_length=4096, num_heads=5):
        super().__init__()

        # ── Shared Sei backbone ──
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))

        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))

        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))

        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, padding=4),
            nn.ReLU(inplace=True))

        # Dilated convolutions
        self.dconv1 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4),
            nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8),
            nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16),
            nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32),
            nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(
            nn.Dropout(p=0.10),
            nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50),
            nn.ReLU(inplace=True))

        # BSpline transformation (reduces spatial dim)
        self._spline_df = 16
        self.spline_tr = nn.Sequential(
            nn.Dropout(p=0.5),
            BSplineTransformation(self._spline_df, scaled=False))

        # ── 5 cell-line-specific heads ──
        # After spline: (B, 960 * 16) features
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(960 * self._spline_df, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        # x: (B, 4, L) — L=4096
        lout1 = self.lconv1(x)
        out1 = self.conv1(lout1)

        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        out = cat_out4 + dconv_out5

        spline_out = self.spline_tr(out)
        reshape_out = spline_out.view(spline_out.size(0), 960 * self._spline_df)

        logits = torch.cat([head(reshape_out) for head in self.heads], dim=1)
        return logits


# ── Data ────────────────────────────────────────────────────────────
def pad_to_4096(seq_np):
    """Center-pad (N, L, 4) OHE array to (N, 4096, 4)."""
    n, l, c = seq_np.shape
    if l == 4096:
        return seq_np
    if l > 4096:
        start = (l - 4096) // 2
        return seq_np[:, start:start + 4096, :]
    pad_total = 4096 - l
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(seq_np, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')


class SeiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, regression=False):
        self.seq = torch.tensor(
            pad_to_4096(data_dict['sequence']).transpose(0, 2, 1),
            dtype=torch.float
        )  # (N, 4, 4096)
        self.labels = torch.tensor(data_dict['label'], dtype=torch.float)
        self.cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long)
        self.regression = regression
        if regression:
            self.y = torch.tensor(data_dict['y'], dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.regression:
            return self.seq[idx], self.y[idx], self.cl_idx[idx]
        return self.seq[idx], self.labels[idx], self.cl_idx[idx]


# ── Training ────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, data_dict, batch_size, device):
    model.eval()
    all_preds = []
    seq = torch.tensor(
        pad_to_4096(data_dict['sequence']).transpose(0, 2, 1),
        dtype=torch.float
    ).to(device)
    cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long, device=device)

    for i in range(0, len(seq), batch_size):
        x_batch = seq[i:i + batch_size]
        cl_batch = cl_idx[i:i + batch_size]
        outputs = model(x_batch)
        preds = outputs[torch.arange(len(cl_batch)), cl_batch]
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Sei — 5-head model (4096bp)')
    parser.add_argument('--device', default='cuda:0', help='GPU device, e.g. cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--check_step', type=int, default=500)
    parser.add_argument('--output_dir', default='./Models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Optional path to pre-existing 4096bp data pickle')
    parser.add_argument('--regression', action='store_true', default=False,
                        help='Train as regression (L1 loss on Logistic_norm targets)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f'[Sei] Device: {device}, Cell lines: {P0_CELL_LINES}')
    print(f'[Sei] Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}')
    print(f'[Sei] Input length: 4096bp (data will be center-padded if needed)')
    print(f'[Sei] Mode: {"regression (L1 loss)" if args.regression else "classification (BCE loss)"}')

    # 1. Data
    print('\n[1/4] Loading data...')
    if args.data_path is not None:
        # Load pre-processed 4096bp data
        print(f'  Loading 4096bp data from {args.data_path}')
        with open(args.data_path, 'rb') as f:
            bundle = pickle.load(f)
        trainset, validset, testset, locusset = bundle[:4] if isinstance(bundle, (list, tuple)) else (bundle, None, None, None)
    else:
        trainset, validset, testset, locusset = build_datasets_from_original()

    train_loader = torch.utils.data.DataLoader(
        SeiDataset(trainset, regression=args.regression),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # 2. Model
    print('\n[2/4] Building model...')
    torch.cuda.empty_cache()
    gc.collect()
    model = Sei5Head(sequence_length=4096, num_heads=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total_params:,}, Trainable: {trainable:,}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # 3. Train
    print('\n[3/4] Training...')
    save_name = f'{args.output_dir}/Sei_5head'
    if args.regression:
        save_name += '_regression'
    best_val_score = -float('inf')
    count = 0
    valid_count = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            if args.regression:
                x, y_target, cl_idx = [b.to(device) for b in batch]
            else:
                x, labels, cl_idx = [b.to(device) for b in batch]

            model.train()
            optimizer.zero_grad()
            outputs = model(x)

            if args.regression:
                loss = F.l1_loss(outputs[torch.arange(len(cl_idx)), cl_idx], y_target)
            else:
                loss = loss_fn(outputs[torch.arange(len(cl_idx)), cl_idx], labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            valid_count += 1

            avg_loss = epoch_loss / n_batches
            print(f"Epoch [{epoch+1}/{args.epochs}] Step {n_batches}, Loss: {avg_loss:.4f}", end='\r', flush=True)

            if valid_count >= args.check_step:
                pv = predict(model, validset, args.batch_size, device)

                if args.regression:
                    # For regression: use negative L1 as validation score
                    y_val = np.asarray(validset['y'], dtype=float)
                    cl_indices = np.asarray(validset['cell_line_idx'], dtype=int)
                    val_scores = []
                    for ci in range(5):
                        mask = cl_indices == ci
                        if mask.sum() > 0:
                            l1 = np.abs(pv[mask] - y_val[mask]).mean()
                            val_scores.append(-l1)
                    val_score = np.mean(val_scores) if val_scores else -float('inf')
                else:
                    results = per_cell_line_metrics(1 / (1 + np.exp(-pv)), validset)
                    valid_results = [r for r in results if not np.isnan(r['AUROC'])]
                    val_score = np.mean([r['AUROC'] for r in valid_results]) if valid_results else 0

                if val_score > best_val_score:
                    best_val_score = val_score
                    count = 0
                    torch.save(model.state_dict(), save_name + '.best.pt')
                    val_metric = "-L1" if args.regression else "AUROC"
                    print(f"\n  Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val mean {val_metric}: {val_score:.4f}  ↑", flush=True)
                else:
                    count += 1
                    val_metric = "-L1" if args.regression else "AUROC"
                    print(f"\n  Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val mean {val_metric}: {val_score:.4f}  -", flush=True)
                    if count >= args.earlystop:
                        val_metric = "-L1" if args.regression else "AUROC"
                        print(f'  Early stop at epoch {epoch+1}, best val {val_metric}: {best_val_score:.4f}', flush=True)
                        break
                valid_count = 0

        torch.save(model.state_dict(), f'{save_name}.epoch{epoch}.pt')
        if count >= args.earlystop:
            break

    best_path = save_name + '.best.pt'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # 4. Evaluate
    print('\n[4/4] Evaluating on test set...')
    pt = predict(model, testset, args.batch_size, device)
    pt_score = pt if args.regression else 1 / (1 + np.exp(-pt))

    results = per_cell_line_metrics(pt_score, testset)
    print_metrics_table(results, 'Test Set')

    mode_str = 'regression (L1 loss)' if args.regression else 'classification (BCE loss)'
    extra = f'Sei 5-head (4096bp input) — {mode_str}\n'
    extra += f'Split: original randomCRE.4.pkl → id-level 7:2:1 + P0 filter\n'
    extra += f'Cell lines: {P0_CELL_LINES}\nEpochs: {args.epochs}, LR: {args.lr}\n'
    save_metrics(results, save_name, extra)

    if locusset is not None and locusset['length'] > 0:
        pl = predict(model, locusset, args.batch_size, device)
        pl_score = pl if args.regression else 1 / (1 + np.exp(-pl))
        locus_results = per_cell_line_metrics(pl_score, locusset)
        print_metrics_table(locus_results, 'Locus Holdout (chr11)')

    print('\nDone!')


if __name__ == '__main__':
    main()
