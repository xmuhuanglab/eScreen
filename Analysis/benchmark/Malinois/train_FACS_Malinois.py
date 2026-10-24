#!/usr/bin/env python3
"""Malinois (BassetBranched) FACS training — single cell line, single head.

What this script implements
---------------------------
The Malinois entry in the FACS readout-specific evaluation of Supplementary
Methods §5.5. Unlike the main benchmark (``train_Malinois.py``), which shares one
backbone across five cell-line heads, the FACS experiment trains a *separate*
model per cell line with a single output head — the FACS datasets for K562 and
hPSC are independent, so there is nothing to share.

Architecture and training recipe follow §6.2.5's Basset adaptation (BassetBranched
backbone, 1000-unit fully connected layer with dropout 0.3, a 1000 -> 64 -> 1 head
with ReLU + dropout 0.2), with ``BCEWithLogitsLoss`` and gradient clipping at norm
1.0. Differences from the manuscript, stated plainly:

* The paper's FACS section specifies AdamW with lr 1e-3 and weight decay 0.01;
  this script defaults to weight decay **1e-5**, the value §6.2.5 uses for the
  main benchmark (the CLI exposes ``--lr`` and ``--epochs`` but not
  ``--weight_decay``).
* Validation runs once per **epoch** and stops after 10 stagnant epochs, whereas
  §6.2.5's main-benchmark run validates every 500 steps. §5.5 only says "stopped
  after 15 epochs without validation improvement"; neither the 10 nor the 15 is
  what this script implements.
* Model selection uses single-cell-line AUROC, while §5.5 selects the eScreen
  checkpoint by the m2a composite.

Expected on disk
----------------
Data comes from ``load_facs_data.load_facs_for_cell_line`` (see below), which
returns ``(trainset, validset, testset)`` dicts with ``sequence`` (one-hot
``(N, L, 4)``), ``label`` and ``length``. No path is named in this file; the
loader's own defaults apply.

Outputs, under ``--output_dir`` (default ``./FACS_benchmark/Malinois``)::

    {output_dir}/{cell_line}/Malinois_FACS.best.pt                 # checkpoint
    {output_dir}/{cell_line}/Malinois_FACS_test_predictions.csv    # per-sample
    {output_dir}/{cell_line}/Malinois_FACS_test_metrics.csv        # AUROC/AUPRC

External dependency not vendored in this repository
---------------------------------------------------
``load_facs_data`` is imported from ``../_shared/``, a directory that does **not**
exist in this repository snapshot; the script cannot run without it. It must
provide ``load_facs_for_cell_line(cell_line)`` returning the three split dicts
described above, with sequences already one-hot encoded as ``(N, L, 4)``.
"""
import argparse, os, sys, gc, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
from load_facs_data import load_facs_for_cell_line


class Basset1Head(nn.Module):
    """BassetBranched backbone + single output head.

    Parameters
    ----------
    input_len : int, optional
        Input length in bp (512).

    Attributes
    ----------
    conv_block : torch.nn.Sequential
        Same three stages as the 5-head benchmark model: 4 -> 300 (k=19, pool 3),
        300 -> 200 (k=11, pool 4), 200 -> 200 (k=7, pool 4), each with batch norm
        and ReLU.
    shared_fc : torch.nn.Sequential
        Flatten -> ``Linear(200 * pooled_len, 1000)`` -> ReLU -> Dropout(0.3).
    head : torch.nn.Sequential
        ``Linear(1000, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 1)``.

    Notes
    -----
    * ``conv_output_len = input_len // 3 // 4 // 4`` reproduces ``MaxPool1d``'s
      floor; for 512 bp that is 10, so the flatten is 2000-wide.
    * ``forward`` flattens the head output to ``(B,)`` — one logit per sequence —
      which is what makes ``BCEWithLogitsLoss`` and ``roc_auc_score`` usable
      directly, without the extra ``squeeze`` the 5-head model would need.
    * The class name says "1Head" rather than "BassetBranched1Head" because it is
      deliberately the 5-head architecture with four heads removed; weights are
      therefore **not** interchangeable with ``BassetBranched5Head`` (different
      parameter names and no ``heads`` ModuleList).
    """

    def __init__(self, input_len=512):
        """Build the shared trunk, the shared FC block and the single head.

        Parameters
        ----------
        input_len : int, optional
            Input length in bp (512). Used only to derive the flatten width
            (``input_len // 3 // 4 // 4``), so it must match the real sequence
            length.

        Notes
        -----
        Splitting the network into ``conv_block`` / ``shared_fc`` / ``head`` (as
        opposed to one flat ``Sequential``) is what allows feature extraction and
        head-swapping experiments without rebuilding the layers.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(4, 300, kernel_size=19, padding=9),
            nn.BatchNorm1d(300), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(300, 200, kernel_size=11, padding=5),
            nn.BatchNorm1d(200), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(200, 200, kernel_size=7, padding=3),
            nn.BatchNorm1d(200), nn.ReLU(), nn.MaxPool1d(4),
        )
        conv_output_len = input_len
        conv_output_len = conv_output_len // 3 // 4 // 4
        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(200 * conv_output_len, 1000),
            nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Sequential(
            nn.Linear(1000, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """Compute one logit per sequence.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA, shape ``(B, 4, L)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` logits (no activation).

        Notes
        -----
        ``self.head(x).flatten()`` collapses ``(B, 1)`` to ``(B,)``; keeping the
        batch dimension intact is essential because a 1-sample batch would
        otherwise become a scalar.
        """
        x = self.conv_block(x)
        x = self.shared_fc(x)
        return self.head(x).flatten()


def prepare_sequence_malinois(data_dict):
    """Convert (N, L, 4) OHE to (N, 4, L) for Conv1d.

    Parameters
    ----------
    data_dict : dict
        Split dict carrying a ``sequence`` array of shape ``(N, L, 4)`` — the
        layout produced by the shared loaders, with the nucleotide axis last.

    Returns
    -------
    torch.Tensor
        ``(N, 4, L)`` float32 tensor, channels first as ``Conv1d`` requires.

    Notes
    -----
    ``np.transpose(0, 2, 1)`` swaps the last two axes and leaves the batch axis
    alone; using ``torch.permute`` directly on a numpy array would not work, and
    transposing axes 1 and 2 (rather than 0 and 1) is exactly the
    channels-last-to-channels-first conversion needed here.

    Transposing a numpy view does not copy, but ``torch.tensor`` always copies,
    so the returned tensor owns its memory and can safely be moved to the GPU.
    """
    seq = data_dict['sequence']
    return torch.tensor(seq.transpose(0, 2, 1), dtype=torch.float)


def train_malinois_facs(cell_line, device='cuda:0', epochs=50, batch_size=64,
                        lr=1e-3, weight_decay=1e-5, earlystop=10,
                        output_dir='./FACS_benchmark/Malinois'):
    """Train Malinois on FACS data for one cell line.

    Parameters
    ----------
    cell_line : {'K562', 'hPSC'}
        Which FACS dataset to train on; also the output subdirectory name.
    device : str, optional
        Torch device string, e.g. ``'cuda:0'``.
    epochs : int, optional
        Maximum epochs.
    batch_size : int, optional
        Training batch size. The default here (64) differs from the CLI default
        (128) and from the 128 that §6.2.5 reports for the main benchmark.
    lr : float, optional
        AdamW learning rate.
    weight_decay : float, optional
        AdamW weight decay; 1e-5 by default (the paper's FACS section says 0.01).
    earlystop : int, optional
        Stop after this many consecutive epochs without an AUROC improvement.
    output_dir : str, optional
        Root for checkpoints and CSVs; a ``{cell_line}`` subdirectory is created.

    Returns
    -------
    dict
        ``{'cell_line', 'model', 'AUROC', 'AUPRC', 'n_pos', 'n_neg'}`` for the
        test split.

    Notes
    -----
    * Shuffling is done by permuting indices and slicing ``trainset['sequence']``
      directly rather than through a ``DataLoader``; that is why the loader dict
      must keep ``sequence``, ``label`` and ``length`` as plain arrays.
    * Validation predictions are accumulated as **probabilities**
      (``torch.sigmoid(...)`` inside the loop) while training works on logits.
    * ``roc_auc_score`` is wrapped in a bare ``except`` that falls back to
      ``val_auroc = 0``: with a single-class validation split (or all-NaN scores)
      sklearn raises, and the epoch is then treated as a non-improvement. The bare
      except would also swallow unrelated errors, so a genuine bug surfaces only
      as "AUROC never improves".
    * ``val_batch_size = min(64, len(val_y))`` avoids an empty-slice edge case
      when the validation split has fewer than 64 rows; ``val_y`` is assigned
      twice (a harmless duplication).
    * The test evaluation is a single forward pass over the whole split — no
      batching — which is fine for a FACS-sized test set but would OOM on a
      genome-wide one.
    * ``AUPRC`` is the trapezoidal ``auc(rec, prec)``, matching the convention used
      elsewhere in this benchmark rather than ``average_precision_score``.
    """
    print(f'\n[1/4] Loading FACS data for {cell_line}...')
    trainset, validset, testset = load_facs_for_cell_line(cell_line)

    save_dir = f'{output_dir}/{cell_line}'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'{save_dir}/Malinois_FACS'

    print(f'\n[2/4] Building model...')
    torch.cuda.empty_cache()
    gc.collect()
    model = Basset1Head(input_len=512).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total:,}, Trainable: {trainable:,}')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    val_seq_all = prepare_sequence_malinois(validset).numpy()
    val_y = validset['label']
    val_batch_size = min(64, len(val_y))
    val_y = validset['label']

    print(f'\n[3/4] Training ({epochs} epochs)...')
    best_val = -float('inf')
    count = 0

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        indices = np.random.permutation(trainset['length'])

        for i in range(0, trainset['length'], batch_size):
            batch_idx = indices[i:i + batch_size]
            x = prepare_sequence_malinois(
                {'sequence': trainset['sequence'][batch_idx]}).to(device)
            y = torch.tensor(trainset['label'][batch_idx], dtype=torch.float, device=device)

            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        all_pv = []
        with torch.no_grad():
            for vi in range(0, len(val_seq_all), val_batch_size):
                vx = torch.tensor(val_seq_all[vi:vi+val_batch_size], dtype=torch.float, device=device)
                # Ensure correct dtype for bf16 models
                all_pv.append(torch.sigmoid(model(vx)).float().cpu().numpy())
        pv = np.concatenate(all_pv)
        try:
            val_auroc = roc_auc_score(val_y, pv)
        except:
            val_auroc = 0

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}  Val AUROC: {val_auroc:.4f}", end='')
        if val_auroc > best_val:
            best_val = val_auroc
            count = 0
            torch.save(model.state_dict(), save_name + '.best.pt')
            print("  (saved)  ↑")
        else:
            count += 1
            print(f"  (best: {best_val:.4f})  -")
            if count >= earlystop:
                print(f'  Early stop at epoch {epoch+1}')
                break

    best_path = save_name + '.best.pt'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    print(f'\n[4/4] Evaluating on test set...')
    test_seq = prepare_sequence_malinois(testset).to(device)
    model.eval()
    with torch.no_grad():
        pt = torch.sigmoid(model(test_seq)).cpu().numpy()

    y_true = testset['label']
    test_auroc = roc_auc_score(y_true, pt)
    prec, rec, _ = precision_recall_curve(y_true, pt)
    test_auprc = auc(rec, prec)

    print(f'\n  === {cell_line} FACS Malinois ===')
    print(f'  AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}')

    pred_df = pd.DataFrame({'true_label': y_true, 'pred_prob': pt})
    pred_df.to_csv(f'{save_name}_test_predictions.csv', index=False)
    metrics = {'cell_line': cell_line, 'model': 'Malinois',
                'AUROC': test_auroc, 'AUPRC': test_auprc,
                'n_pos': int(y_true.sum()), 'n_neg': int((1 - y_true).sum())}
    pd.DataFrame([metrics]).to_csv(f'{save_name}_test_metrics.csv', index=False)
    return metrics


def main():
    """Parse CLI arguments and train one cell line's FACS model.

    Returns
    -------
    None

    Notes
    -----
    Only ``K562`` and ``hPSC`` are accepted (``choices=``), matching the two FACS
    datasets of Supplementary Methods §5.5. To reproduce the manuscript's FACS
    comparison, run this script once per cell line (there is no loop over cell
    lines) and then do the same for every other baseline.
    """
    parser = argparse.ArgumentParser(description='Malinois FACS training')
    parser.add_argument('--cell_line', default='K562', choices=['K562', 'hPSC'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--output_dir', default='./FACS_benchmark/Malinois')
    args = parser.parse_args()
    metrics = train_malinois_facs(
        cell_line=args.cell_line, device=args.device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, earlystop=args.earlystop,
        output_dir=args.output_dir)
    print(f'\nDone! {args.cell_line}: AUROC={metrics["AUROC"]:.4f}')


if __name__ == '__main__':
    main()