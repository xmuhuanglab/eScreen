#!/usr/bin/env python3
"""
Malinois (BassetBranched) — 5-head model for 5 cell lines, 512bp input.

Usage:
    python train_Malinois.py --device cuda:0 --epochs 50 --batch_size 128

What this script implements
---------------------------
The Malinois arm of Supplementary Methods §6.2.5. Malinois is a Basset-derived
CNN for MPRA-based CRE activity; for the eScreen benchmark its single regression
output is replaced by five cell-line-specific heads (A375, HT29, HepG2, K562,
hPSC), and only the head matching each sample's cell line is supervised. The
architecture matches the manuscript exactly: Conv1d 4->300 (k=19, pool 3),
300->200 (k=11, pool 4), 200->200 (k=7, pool 4), each with batch norm and ReLU,
then a flattened 1000-unit layer with dropout 0.3, then the per-cell-line
two-layer heads (1000 -> 64 -> 1, ReLU + dropout 0.2 between them).

Training configuration also follows §6.2.5: 512 bp one-hot input, AdamW
(lr 1e-3, weight decay 1e-5), ``BCEWithLogitsLoss``, batch size 128, gradient
clipping at norm 1.0, validation every 500 steps on the mean AUROC across the five
cell lines, early stopping after 10 stagnant evaluations, best checkpoint
restored.

Expected on disk
----------------
Data comes from ``data_utils`` (see below), which reads the original eScreen
bundle and applies the P0 filter + identifier-level 7:2:1 split; this script does
not itself name any data path. Outputs, relative to the working directory::

    ./Models/Malinois_5head.best.pt        # best validation checkpoint
    ./Models/Malinois_5head.epoch{N}.pt    # snapshot after every epoch
    ./Models/Malinois_5head_test_results.csv   # via save_metrics

External dependency not vendored in this repository
---------------------------------------------------
``data_utils`` is imported from ``../_shared/``, a directory that does **not**
exist in this repository snapshot. It provides
``build_datasets_from_original`` (returning the ``trainset``/``validset``/
``testset``/``locusset`` dicts), ``prepare_sequence_1d`` (one-hot ``(N, L, 4)`` ->
channels-first tensor), ``per_cell_line_metrics``, ``print_metrics_table``,
``save_metrics``, ``P0_CELL_LINES`` and ``CELL_LINE_TO_IDX``. The script cannot run
without that module. ``per_cell_line_metrics`` here takes ``(probs, data_dict)`` —
a different signature from the same-named helper in ``alphagenome_common``.
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
    """BassetBranched backbone with 5 separate output heads (one per cell line).

    Parameters
    ----------
    input_len : int, optional
        Input length in bp (512 for the eScreen benchmark).

    Attributes
    ----------
    conv_block : torch.nn.Sequential
        Shared convolutional trunk: 4 -> 300 -> 200 -> 200 channels with kernel
        sizes 19/11/7 and max-pooling 3/4/4.
    shared_fc : torch.nn.Sequential
        Flatten -> ``Linear(200 * pooled_len, 1000)`` -> ReLU -> Dropout(0.3).
    heads : torch.nn.ModuleList
        Five identical ``Linear(1000, 64) -> ReLU -> Dropout(0.2) ->
        Linear(64, 1)`` blocks, in ``P0_CELL_LINES`` order (A375, HT29, HepG2,
        K562, hPSC).

    Notes
    -----
    * With ``input_len=512`` the pooled length is ``512 // 3 // 4 // 4 == 10``, so
      the flatten feeds ``200 * 10 = 2000`` units into the 1000-unit layer.
      Integer division must mirror ``MaxPool1d``'s floor, otherwise the linear
      layer's ``in_features`` will not match the flattened tensor.
    * ``forward`` concatenates all five heads, returning ``(B, 5)`` — one logit
      per cell line per sample. Supervision is applied only to the column
      matching the sample's own cell line (see :func:`train_epoch`), so the other
      four heads receive no gradient for that sample. This is the manuscript's
      "only the head corresponding to the cell line that each sample belonged to
      was supervised".
    * Shared backbone + per-line heads is what lets the model pool statistical
      strength across cell lines while keeping decision boundaries separate.
    """

    def __init__(self, input_len=512):
        """Build the shared backbone and the five cell-line heads.

        Parameters
        ----------
        input_len : int, optional
            Input length in bp (512); sets the flatten width through
            ``input_len // 3 // 4 // 4``.

        Notes
        -----
        All five heads are structurally identical and initialised independently;
        they differ only in which samples' gradients reach them. There is no
        parameter sharing between heads — only the backbone is shared, as in
        Supplementary Methods §6.2.5.
        """
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
        """Compute the five per-cell-line logits for a batch.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA, shape ``(B, 4, L)``, ``L = input_len``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 5)``; column *j* is the logit for ``P0_CELL_LINES[j]``.
            No activation — the loss is applied to a gathered column.
        """
        # x: (B, 4, L)
        x = self.conv_block(x)
        x = self.shared_fc(x)
        logits = torch.cat([head(x) for head in self.heads], dim=1)  # (B, 5)
        return logits


# ── Training ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, loss_fn):
    """Run one pass over a DataLoader and return the mean batch loss.

    Parameters
    ----------
    model : BassetBranched5Head
        Model, switched to train mode by the caller's loop (this function sets it
        too, so calling it directly is safe).
    loader : torch.utils.data.DataLoader
        Yields ``(x, labels, cl_idx)``.
    optimizer : torch.optim.Optimizer
        AdamW.
    device : torch.device
        Where batches are moved.
    loss_fn : callable
        ``nn.BCEWithLogitsLoss()``.

    Returns
    -------
    float
        Mean batch loss; ``0.0`` when the loader yields nothing (guarded by
        ``max(n_batches, 1)``).

    Notes
    -----
    * **Selective supervision**: ``outputs[arange(B), cl_idx]`` gathers, for each
      sample, the logit from its own cell-line head. The other four columns are
      computed but not differentiated through, which is why the five heads can
      coexist in one model without cross-cell-line interference.
    * Gradient clipping at norm 1.0 follows §6.2.5.
    * This function is **not used** by :func:`main`, whose training loop is
      inlined so that it can perform step-counted validation inside the batch
      loop. It is kept as the reusable/clean version; changing it will not change
      the script's results.
    """
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
    """Return predictions (logits) for the matching head of each sample.

    Parameters
    ----------
    model : BassetBranched5Head
        Model; switched to ``eval()`` for the duration.
    data_dict : dict
        A split dict from ``data_utils`` with ``sequence`` (one-hot ``(N, L, 4)``),
        ``cell_line_idx`` and ``label``.
    batch_size : int
        Inference batch size.
    device : torch.device
        Target device.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)`` logits, gathered from each sample's own cell-line head.

    Notes
    -----
    * The whole split is one-hot encoded and moved to the GPU up front
      (``prepare_sequence_1d(data_dict, seq_len=512).to(device)``) — fine for the
      few hundred thousand 512 bp constructs at 4 channels but the dominant
      memory cost of the script.
    * Returns *logits*; every call site applies ``1 / (1 + exp(-x))`` itself
      rather than ``torch.sigmoid``, which is numerically equivalent but slightly
      less stable for large negative values (``exp`` overflow rather than a
      clamped sigmoid).
    * ``model.eval()`` matters here: BatchNorm would otherwise use batch
      statistics, and with a small tail batch the predictions would shift.
    """
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
    """Map-style dataset over a ``data_utils`` split dict.

    Parameters
    ----------
    data_dict : dict
        Split dict with ``sequence`` (one-hot ``(N, L, 4)``), ``label`` and
        ``cell_line_idx``.
    seq_len : int, optional
        Length passed to ``prepare_sequence_1d``; should equal ``L`` (512 here).

    Attributes
    ----------
    seq : torch.Tensor
        ``(N, 4, L)`` float tensor — the one-hot encoding is done **once** in the
        constructor, not per ``__getitem__``.
    labels : torch.Tensor
        ``(N,)`` float32 labels.
    cl_idx : torch.Tensor
        ``(N,)`` int64 cell-line indices.

    Notes
    -----
    The eager encoding is the right trade-off here (the split is materialised in
    RAM anyway and DataLoader workers would each pay the conversion cost), but it
    means the dataset cannot be used lazily for a split that does not fit in
    memory.
    """

    def __init__(self, data_dict, seq_len=512):
        """One-hot encode the split and move it into torch tensors.

        Parameters
        ----------
        data_dict : dict
            Split dict from ``data_utils`` with ``sequence``, ``label`` and
            ``cell_line_idx``.
        seq_len : int, optional
            Forwarded to ``prepare_sequence_1d``; 512 for this benchmark.

        Notes
        -----
        The ``(N, L, 4) -> (N, 4, L)`` transposition happens inside
        ``prepare_sequence_1d``, not here, so ``self.seq`` is already
        channels-first and ``__getitem__`` is a pure slice.
        """
        self.seq = prepare_sequence_1d(data_dict, seq_len=seq_len)
        self.labels = torch.tensor(data_dict['label'], dtype=torch.float)
        self.cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long)

    def __len__(self):
        """Number of samples in the split.

        Returns
        -------
        int
            ``len(self.labels)``.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """Return one ``(sequence, label, cell_line_idx)`` triple.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple
            ``(seq[idx], labels[idx], cl_idx[idx])`` with shapes ``(4, L)``,
            ``()`` and ``()``.
        """
        return self.seq[idx], self.labels[idx], self.cl_idx[idx]


# ── Main ────────────────────────────────────────────────────────────
def main():
    """Train the 5-head Malinois model and report test / locus metrics.

    Returns
    -------
    None

    Notes
    -----
    * The training loop is **inlined** here rather than calling
      :func:`train_epoch`, because validation must fire every
      ``args.check_step`` (500) *batches* — mid-epoch — to match the manuscript's
      "every 500 steps" schedule. ``valid_count`` therefore counts batches since
      the last check, not epochs.
    * ``drop_last=True`` on the training loader keeps BatchNorm statistics stable
      (a one-sample tail batch would make them meaningless); the cost is that the
      remainder of each epoch is dropped.
    * Early stopping decrements patience only on validation checks, and the
      ``break`` inside the batch loop is followed by a second ``break`` after the
      epoch checkpoint save, so the loop exits cleanly.
    * An epoch snapshot is written after every epoch
      (``Malinois_5head.epoch{N}.pt``) *in addition to* the best checkpoint, so
      disk usage grows with ``--epochs``.
    * The best checkpoint is reloaded with ``map_location=device`` and, if no
      checkpoint exists at all (e.g. ``--check_step`` larger than the dataset),
      evaluation silently proceeds with the last-epoch weights.
    * The locus holdout is reported but never used for model selection; it is
      ``None``-guarded because ``data_utils`` may return an empty split.
    """
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
