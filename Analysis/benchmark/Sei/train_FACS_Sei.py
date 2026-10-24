#!/usr/bin/env python3
"""
Sei FACS training — single cell line, single output head, 4096bp input.
Sei backbone (BSplineConv1D + dilated convs) + single head.

FACS readout-specific evaluation of the Sei baseline, i.e. the Sei analogue of
Supplementary Methods §5.5. One model is trained per FACS cell line (``K562``
or ``hPSC``) on clinical-trial-style held-out splits, with 512 bp one-hot CRE
windows centre-padded to Sei's native 4,096 bp width.

Attribution caveat
------------------
The manuscript never describes a Sei baseline: §6.2 covers Enformer,
AlphaGenome, HyenaDNA, LegNet/Dream and DeepSEA/NvwaCE/Malinois only. The model
definition, padding width, head width and hyper-parameters here are local
choices made to mirror the other FACS scripts; the evaluation protocol
(tRRA-thresholded labels, per-cell-line AUROC/AUPRC) follows §5.5 and §7.1.

Model wrapped
-------------
``Sei1Head``, an inline copy of the Sei trunk from ``Sei.Sei_Default`` (that
class cannot be instantiated -- see ``Sei.py``) plus one head. Only
``BSplineTransformation`` is imported from the vendored module. Trained from
scratch; no Sei pretrained weights exist in this repository.

Expected on disk
----------------
* ``../_shared/load_facs_data.py`` -- ``load_facs_for_cell_line``.
  **Missing from this checkout**, so the script does not run as shipped. It
  returns ``(trainset, validset, testset)`` dicts with ``'sequence'``
  ``(N, 512, 4)``, ``'label'`` ``(N,)`` and ``'length'``; labels follow §5.5
  (positive when tRRA > 2).

Outputs (``--output_dir``, default ``./FACS_benchmark/Sei``)::

    <output_dir>/<cell_line>/Sei_FACS.best.pt
    <output_dir>/<cell_line>/Sei_FACS_test_predictions.csv
    <output_dir>/<cell_line>/Sei_FACS_test_metrics.csv

Entry point
-----------
    python train_FACS_Sei.py --cell_line K562 --device cuda:0

Deviations from §5.5 (code, not paper)
-------------------------------------
Weight decay defaults to 1e-7 (the paper says 0.01 for the FACS experiment),
validation runs once per epoch rather than every 500 steps, model selection uses
validation AUROC instead of the m2a composite score, and there is no boosted
sampler: the script trains with plain ``BCEWithLogitsLoss`` and the batch is
assembled by a plain permutation.
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
from Sei import BSplineTransformation
from load_facs_data import load_facs_for_cell_line


# ── Model ───────────────────────────────────────────────────────────
class Sei1Head(nn.Module):
    """Sei trunk with a single scalar output head.

    Identical trunk to ``train_Sei.Sei5Head`` (three conv stages, five dilated
    residual convolutions, B-spline compression to 960 * 16 features), with one
    ``Linear(960 * 16, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 1)`` head
    because only one cell line is modelled.

    Parameters
    ----------
    sequence_length : int, default 4096
        Declared input width; no layer is sized from it, but the spline basis
        assumes the length is divisible by 64.

    Notes
    -----
    Shape flow: ``(B, 4, 4096)`` -> ``(B, 960, 64)`` -> spline -> ``(B, 15360)``
    -> one logit. The trunk is duplicated from ``train_Sei.py`` and from
    ``Sei.Sei_Default``; all three copies must be edited together.
    """

    def __init__(self, sequence_length=4096):
        """Build the trunk layers and the single head.

        Parameters
        ----------
        sequence_length : int, default 4096
            Declared input width; informational only.
        """
        super().__init__()
        # Shared Sei backbone
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 480, kernel_size=9, padding=4),
            nn.Conv1d(480, 480, kernel_size=9, padding=4))
        self.conv1 = nn.Sequential(
            nn.Conv1d(480, 480, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(480, 480, kernel_size=9, padding=4), nn.ReLU(inplace=True))
        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(p=0.2),
            nn.Conv1d(480, 640, kernel_size=9, padding=4),
            nn.Conv1d(640, 640, kernel_size=9, padding=4))
        self.conv2 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(640, 640, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(640, 640, kernel_size=9, padding=4), nn.ReLU(inplace=True))
        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4), nn.Dropout(p=0.2),
            nn.Conv1d(640, 960, kernel_size=9, padding=4),
            nn.Conv1d(960, 960, kernel_size=9, padding=4))
        self.conv3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv1d(960, 960, kernel_size=9, padding=4), nn.ReLU(inplace=True),
            nn.Conv1d(960, 960, kernel_size=9, padding=4), nn.ReLU(inplace=True))
        # Dilated convolutions
        self.dconv1 = nn.Sequential(nn.Dropout(p=0.10), nn.Conv1d(960, 960, kernel_size=5, dilation=2, padding=4), nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(nn.Dropout(p=0.10), nn.Conv1d(960, 960, kernel_size=5, dilation=4, padding=8), nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(nn.Dropout(p=0.10), nn.Conv1d(960, 960, kernel_size=5, dilation=8, padding=16), nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(nn.Dropout(p=0.10), nn.Conv1d(960, 960, kernel_size=5, dilation=16, padding=32), nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(nn.Dropout(p=0.10), nn.Conv1d(960, 960, kernel_size=5, dilation=25, padding=50), nn.ReLU(inplace=True))
        # BSpline
        self._spline_df = 16
        self.spline_tr = nn.Sequential(nn.Dropout(p=0.5), BSplineTransformation(self._spline_df, scaled=False))
        # Single output head
        self.head = nn.Sequential(
            nn.Linear(960 * self._spline_df, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 1))

    def forward(self, x):
        """Return one logit per sample.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA ``(B, 4, 4096)``.

        Returns
        -------
        torch.Tensor
            ``(B,)`` raw logits (``flatten()`` removes the trailing 1).
        """
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
        return self.head(reshape_out).flatten()


# ── Data ────────────────────────────────────────────────────────────
def pad_to_4096(seq_np):
    """Centre-pad or centre-crop a one-hot array to exactly 4,096 bp.

    Parameters
    ----------
    seq_np : numpy.ndarray
        One-hot DNA ``(N, L, 4)``.

    Returns
    -------
    numpy.ndarray
        ``(N, 4096, 4)`` of the same dtype; zero-padded when shorter, cropped
        from the centre when longer.

    Notes
    -----
    Zero padding creates all-zero columns, which is *not* the same as padding
    with N. Byte-for-byte duplicate of the helper in ``train_Sei.py``.
    """
    n, l, c = seq_np.shape
    if l == 4096: return seq_np
    if l > 4096:
        start = (l - 4096) // 2
        return seq_np[:, start:start + 4096, :]
    pad_total = 4096 - l
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(seq_np, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant')


def train_sei_facs(cell_line, device='cuda:0', epochs=50, batch_size=32,
                   lr=1e-3, weight_decay=1e-7, earlystop=10,
                   output_dir='./FACS_benchmark/Sei'):
    """Train Sei on the FACS readout of a single cell line.

    Parameters
    ----------
    cell_line : {'K562', 'hPSC'}
        FACS cell line; one independent model per line.
    device : str, default 'cuda:0'
        Torch device string (hard-coded default).
    epochs : int, default 50
        Maximum number of epochs.
    batch_size : int, default 32
        Training batch size, as for the other Sei runs (the 4,096 bp input
        makes larger batches expensive).
    lr : float, default 1e-3
        AdamW learning rate (matches §5.5).
    weight_decay : float, default 1e-7
        AdamW weight decay. §5.5 reports 0.01 for the FACS experiment, so this
        default does not reproduce the paper.
    earlystop : int, default 10
        Epochs without validation-AUROC improvement before stopping.
    output_dir : str, default './FACS_benchmark/Sei'
        Root directory; artefacts go to ``<output_dir>/<cell_line>/``.

    Returns
    -------
    dict
        ``{'cell_line', 'model', 'AUROC', 'AUPRC', 'n_pos', 'n_neg'}`` on the
        held-out test split.

    Notes
    -----
    Plain ``BCEWithLogitsLoss``, global gradient clipping at norm 1.0, no
    class-balanced sampler, and epoch-level rather than step-level validation
    -- all simplified relative to §5.5.
    """
    print(f'\n[1/4] Loading FACS data for {cell_line}...')
    trainset, validset, testset = load_facs_for_cell_line(cell_line)

    save_dir = f'{output_dir}/{cell_line}'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'{save_dir}/Sei_FACS'

    print(f'\n[2/4] Building model...')
    torch.cuda.empty_cache()
    gc.collect()
    model = Sei1Head(sequence_length=4096).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total:,}, Trainable: {trainable:,}')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f'\n[3/4] Training...')
    best_val = -float('inf')
    count = 0
    valid_count = 0

    # Validation tensors (keep on CPU, send in batches)
    # Validation (and later test) windows are padded once and kept in host memory,
    # then copied to the device batch by batch. At 4,096 bp x 4 channels this is
    # 64 KB per sample in float32, which is why the full padded training split is
    # never held as a single tensor.
    val_seq_all = pad_to_4096(validset['sequence']).transpose(0, 2, 1).astype(np.float32)
    val_y = validset['label']
    # ``val_batch_size`` is reused for the test pass below, under the assumption
    # that both splits are of comparable size.
    val_batch_size = min(32, len(val_y))

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        indices = np.random.permutation(trainset['length'])
        for i in range(0, trainset['length'], batch_size):
            batch_idx = indices[i:i + batch_size]
            # Padding is redone for every batch (and every epoch) rather than once
            # for the whole split: it trades host memory for CPU time, which is the
            # cheaper side of the trade at this batch size.
            x_np = pad_to_4096(trainset['sequence'][batch_idx]).transpose(0, 2, 1)
            x = torch.tensor(x_np, dtype=torch.float, device=device)
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
            valid_count += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation (batched to avoid OOM)
        model.eval()
        all_pv = []
        with torch.no_grad():
            for vi in range(0, len(val_seq_all), val_batch_size):
                vx = torch.tensor(val_seq_all[vi:vi+val_batch_size], dtype=torch.float, device=device)
                all_pv.append(torch.sigmoid(model(vx)).cpu().numpy())
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
    # Test evaluation (batched)
    test_seq_all = pad_to_4096(testset['sequence']).transpose(0, 2, 1).astype(np.float32)
    model.eval()
    all_pt = []
    with torch.no_grad():
        for ti in range(0, len(test_seq_all), val_batch_size):
            tx = torch.tensor(test_seq_all[ti:ti+val_batch_size], dtype=torch.float, device=device)
            all_pt.append(torch.sigmoid(model(tx)).cpu().numpy())
    pt = np.concatenate(all_pt)

    y_true = testset['label']
    test_auroc = roc_auc_score(y_true, pt)
    prec, rec, _ = precision_recall_curve(y_true, pt)
    test_auprc = auc(rec, prec)

    print(f'\n  === {cell_line} FACS Sei ===')
    print(f'  AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}')

    pred_df = pd.DataFrame({'true_label': y_true, 'pred_prob': pt})
    pred_df.to_csv(f'{save_name}_test_predictions.csv', index=False)
    metrics = {'cell_line': cell_line, 'model': 'Sei',
               'AUROC': test_auroc, 'AUPRC': test_auprc,
               'n_pos': int(y_true.sum()), 'n_neg': int((1 - y_true).sum())}
    pd.DataFrame([metrics]).to_csv(f'{save_name}_test_metrics.csv', index=False)
    return metrics


def main():
    """Parse CLI arguments, run :func:`train_sei_facs`, print the test AUROC."""
    parser = argparse.ArgumentParser(description='Sei FACS training')
    parser.add_argument('--cell_line', default='K562', choices=['K562', 'hPSC'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--output_dir', default='./FACS_benchmark/Sei')
    args = parser.parse_args()
    metrics = train_sei_facs(
        cell_line=args.cell_line, device=args.device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, earlystop=args.earlystop,
        output_dir=args.output_dir)
    print(f'\nDone! {args.cell_line}: AUROC={metrics["AUROC"]:.4f}')

if __name__ == '__main__':
    main()
