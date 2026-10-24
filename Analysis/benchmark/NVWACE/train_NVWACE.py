#!/usr/bin/env python3
"""
NVWACE (ResNeXt34) — 5-head model for 5 cell lines, 512bp input.

Usage:
    python train_NVWACE.py --device cuda:0 --epochs 50 --batch_size 64

What this script does
---------------------
Implements the NVWACE ("NvwaCE", ref. 53) entry of the per-model
benchmark in Supplementary Methods §6.2.5: a ResNeXt-34 trunk over 512 bp
one-hot DNA with a shared 1,024 -> 256 feature layer and five independent
cell-line heads (A375, HT29, HepG2, K562, hPSC). It is a *baseline*: no
eScreen component is used and no NvwaCE pretrained weights are loaded, so
the network is trained from scratch on the eScreen benchmark dataset.

External model wrapped
----------------------
``ResNeXt_conv1_128_btnk_2dense.ResNeXt`` (vendored, see that module's
docstring). Only the trunk up to the global average pool is kept; the
NvwaCE bottleneck/classifier is bypassed -- see :class:`ResNeXt5Head`.

Expected on disk
----------------
* ``../_shared/data_utils.py`` -- dataset construction, per-cell-line
  metrics and CSV writers. **This module is not present in the current
  checkout** (``Analysis/benchmark/_shared`` does not exist), so the script
  only runs where the authors kept the shared helpers. It is also the
  module that defines ``P0_CELL_LINES`` and ``CELL_LINE_TO_IDX``.
* Input data: ``randomCRE.4.pkl`` as referenced by
  ``build_datasets_from_original()`` (that path is hard-coded inside the
  missing helper, not here). Splitting is identifier-level 7:2:1 with the
  P0 cell-line filter (§6.1, §2.3.3).

Outputs, under ``--output_dir`` (default ``./Models``)::

    Models/NVWACE_5head.best.pt        # best mean-validation-AUROC weights
    Models/NVWACE_5head.epoch<N>.pt    # per-epoch snapshot (all epochs)
    Models/NVWACE_5head_metrics.csv    # written by _shared.save_metrics

plus the per-cell-line metric table printed to stdout, for both the test
split and the chr11 locus holdout (§2.3.3).

Hyper-parameters vs Supplementary Methods §6.2.5
-----------------------------------------------
These defaults reproduce the text exactly: AdamW, lr 1e-3, weight decay
1e-5, batch 64, ``BCEWithLogitsLoss``, gradient clipping at norm 1.0,
validation every 500 steps, early-stopping patience 10, best checkpoint
restored before testing. Two implementation choices are not discussed in
the paper: (i) the shared feature layer and heads use BatchNorm1d, whose
batch statistics make small final batches unstable -- ``drop_last=True``
is therefore load-bearing, not cosmetic; (ii) only the head belonging to
a sample's cell line is supervised (the other four receive no gradient),
which is the same convention §6.2.5 states for Malinois.

Entry point
-----------
    python train_NVWACE.py --device cuda:0 --epochs 50 --batch_size 64
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

# Import ResNeXt building blocks
from ResNeXt_conv1_128_btnk_2dense import ResNeXt, BasicBlock, DenseLayer


# ── Model ───────────────────────────────────────────────────────────
class ResNeXt5Head(nn.Module):
    """ResNeXt-34 trunk with a shared feature layer and five cell-line heads.

    Layout follows Supplementary Methods §6.2.5: the 1,024-d trunk feature
    is projected to 256 with BatchNorm + GELU + dropout 0.2, and each head is
    a 256 -> 32 -> 1 MLP with BatchNorm, GELU and dropout 0.1 between the two
    linear layers. Heads are *not* combined into one output; a sample is
    scored by the head of its own cell line.

    Parameters
    ----------
    num_heads : int, default 5
        Number of cell lines. The order must match ``P0_CELL_LINES`` in the
        shared data helpers: A375, HT29, HepG2, K562, hPSC.

    Attributes
    ----------
    backbone : ResNeXt
        Trunk only -- its own bottleneck/classifier are never called here.
    shared : nn.Sequential
        ``Linear(1024, 256) -> BatchNorm1d -> GELU -> Dropout(0.2)``.
    heads : nn.ModuleList
        Five ``Linear(256, 32) -> BatchNorm1d -> GELU -> Dropout(0.1) ->
        Linear(32, 1)`` stacks.
    """

    def __init__(self, num_heads=5):
        """Build the trunk wrapper and the heads.

        Parameters
        ----------
        num_heads : int, default 5
            Number of independent cell-line heads.
        """
        super().__init__()
        # Shared ResNeXt34 backbone (without original classifier)
        # Building the full ``ResNeXt`` (including its NvwaCE bottleneck, fc1 and
        # fc2) and then ignoring those layers keeps the trunk definition in one
        # place, at the cost of dead parameters in the state dict. Their weights
        # are never updated and their values are irrelevant to the metrics.
        self.backbone = ResNeXt(BasicBlock, [3, 4, 6, 3], num_classes=num_heads)

        # Replace the original fc layers with a shared feature extractor + 5 heads
        # Get the feature dimension from backbone (after avgpool)
        # The backbone outputs 1024 features after layer4 + avgpool
        # Original: bottleneck(1024→32) → fc1(32→num_heads) → Sigmoid
        # New: bottleneck(1024→32) → shared → 5 heads
        # ``self.bottleneck`` / ``self.fc1`` / ``self.fc2`` are bound but unused:
        # :meth:`forward` goes straight from ``avgpool`` to ``self.shared``.
        self.bottleneck = self.backbone.bottleneck
        self.fc1 = self.backbone.fc1  # We'll bypass this
        self.fc2 = self.backbone.fc2  # We'll bypass this

        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # 5 cell-line-specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 32),
                nn.BatchNorm1d(32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        """Return one logit per cell line for every sample.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, 4, 1, L)`` one-hot DNA; a 3-D ``(B, 4, L)`` input gets the
            height axis inserted.

        Returns
        -------
        torch.Tensor
            ``(B, num_heads)`` raw logits. The caller selects the column
            matching each sample's cell line; always apply this selection
            *before* any sigmoid so that ``BCEWithLogitsLoss`` stays stable.
        """
        # x: (B, 4, 1, L)
        if len(x.shape) == 3:
            x = x.unsqueeze(2)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)  # (B, 1024)

        x = self.shared(x)
        logits = torch.cat([head(x) for head in self.heads], dim=1)  # (B, 5)
        return logits


# ── Data ────────────────────────────────────────────────────────────
def prepare_sequence_resnext(data_dict):
    """Convert ``(N, L, 4)`` one-hot DNA to the ``(N, 4, 1, L)`` 2-D layout.

    Parameters
    ----------
    data_dict : dict
        Must contain ``'sequence'`` of shape ``(N, L, 4)``: samples first,
        base position second, A/C/G/T channel last.

    Returns
    -------
    torch.Tensor
        ``float32`` tensor ``(N, 4, 1, L)``. The channel axis moves to
        position 1 and a singleton height axis is inserted at position 2,
        which is what the ``(1, k)`` convolutions of the trunk expect.

    Notes
    -----
    The transpose returns a non-contiguous view; ``torch.tensor(...)``
    copies it, so the result is contiguous but doubles peak memory for the
    split during the conversion. Identical logic is duplicated in
    ``train_FACS_NVWACE.py``; keep the two in sync.
    """
    seq = data_dict['sequence']  # (N, L, 4)
    seq = seq.transpose(0, 2, 1)  # (N, 4, L)
    return torch.tensor(seq[:, :, None, :], dtype=torch.float)  # (N, 4, 1, L)


class EScreenDataset(torch.utils.data.Dataset):
    """Map-style dataset over a pre-built split dict.

    The whole split is materialised in memory as a single
    ``(N, 4, 1, 512)`` float tensor rather than converted per item; at
    512 bp this is 8 KB per sample, so a ~145k-sample training split needs
    roughly 1.2 GB of host RAM.

    Parameters
    ----------
    data_dict : dict
        Split with ``'sequence'`` ``(N, 512, 4)`` one-hot, ``'label'``
        ``(N,)`` in {0, 1} and ``'cell_line_idx'`` ``(N,)`` in 0..4.
    """
    def __init__(self, data_dict):
        """Materialise a split as tensors.

        Parameters
        ----------
        data_dict : dict
            Split dictionary; see the class docstring.
        """
        self.seq = prepare_sequence_resnext(data_dict)
        self.labels = torch.tensor(data_dict['label'], dtype=torch.float)
        self.cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long)

    def __len__(self):
        """Number of samples in the split."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return ``(sequence, label, cell_line_index)`` for one sample.

        Returns
        -------
        tuple of torch.Tensor
            ``(4, 1, L)`` float, scalar float, scalar int64.
        """
        return self.seq[idx], self.labels[idx], self.cl_idx[idx]


# ── Training ────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, data_dict, batch_size, device):
    """Score a whole split with the head matching each sample's cell line.

    Parameters
    ----------
    model : ResNeXt5Head
        Model in any mode; ``eval()`` is called here so BatchNorm uses
        running statistics and dropout is disabled.
    data_dict : dict
        Split with ``'sequence'`` and ``'cell_line_idx'``.
    batch_size : int
        Inference batch size (reuses the training value).
    device : torch.device
        Device the model lives on.

    Returns
    -------
    numpy.ndarray
        ``float32`` logits of shape ``(N,)``, one scalar per sample, in
        input order. Sigmoid is *not* applied.

    Notes
    -----
    The whole split is moved to the GPU up front, so peak device memory is
    ``4 bytes x N x 4 x L`` (about 1.2 GB for the 145k-sample training
    split); validation and test splits are small enough for this to be
    cheaper than per-batch transfers.
    """
    model.eval()
    all_preds = []
    seq = prepare_sequence_resnext(data_dict).to(device)
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
    """Parse CLI arguments, train the 5-head model, evaluate and save.

    Returns
    -------
    None
        Everything is written to ``--output_dir``; the test and locus
        metric tables are printed.
    """
    parser = argparse.ArgumentParser(description='NVWACE — 5-head ResNeXt34')
    parser.add_argument('--device', default='cuda:0', help='GPU device, e.g. cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--check_step', type=int, default=500)
    parser.add_argument('--output_dir', default='./Models')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f'[NVWACE] Device: {device}, Cell lines: {P0_CELL_LINES}')
    print(f'[NVWACE] Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}')

    # 1. Data
    print('\n[1/4] Loading data...')
# Four splits come back from the shared helper: train/valid/test built by
# identifier-level 7:2:1 (§6.1) plus an extra chr11 locus holdout used for
# the leakage-controlled check of §2.3.3. The locus split may be absent, in
# which case it is skipped at the end of this function.
    trainset, validset, testset, locusset = build_datasets_from_original()

    train_loader = torch.utils.data.DataLoader(
        EScreenDataset(trainset),
        # ``drop_last=True`` is required, not optional: BatchNorm1d in the
        # shared layer and heads cannot compute batch statistics from a
        # singleton final batch, which would raise on the last step of an epoch.
        # The cost is that up to ``batch_size - 1`` samples are dropped per
        # epoch.
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # 2. Model
    print('\n[2/4] Building model...')
    torch.cuda.empty_cache()
    gc.collect()
    model = ResNeXt5Head(num_heads=5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total_params:,}, Trainable: {trainable:,}')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # 3. Train
    print('\n[3/4] Training...')
    save_name = f'{args.output_dir}/NVWACE_5head'
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

            if valid_count >= args.check_step:
                pv = predict(model, validset, args.batch_size, device)
                # The model emits logits, so the sigmoid is applied here before
                # AUROC/AUPRC -- the convention stated in §6.1 ("AUROC, calculated
                # after applying a sigmoid function to the output logits"). Evaluated
                # with ``np.exp`` on float64 rather than ``torch.sigmoid``; the two
                # agree to float precision for |logit| well below ~700.
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

# Per-epoch snapshot regardless of validation outcome; the ``.best.pt``
# file written inside the step loop remains the checkpoint actually
# restored below.
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

    extra = f'NVWACE (ResNeXt34) 5-head\nSplit: original randomCRE.4.pkl → id-level 7:2:1 + P0 filter\n'
    extra += f'Cell lines: {P0_CELL_LINES}\nEpochs: {args.epochs}, LR: {args.lr}\n'
    save_metrics(results, save_name, extra)

    if locusset is not None and locusset['length'] > 0:
        pl = predict(model, locusset, args.batch_size, device)
        pl_sigmoid = 1 / (1 + np.exp(-pl))
        locus_results = per_cell_line_metrics(pl_sigmoid, locusset)
        print_metrics_table(locus_results, 'Locus Holdout (chr11)')

    print('\nDone!')


if __name__ == '__main__':
    main()
