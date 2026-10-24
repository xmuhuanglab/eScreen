#!/usr/bin/env python3
"""NVWACE (ResNeXt34) FACS training -- single cell line, single head.

FACS readout-specific evaluation of the NVWACE baseline (Supplementary
Methods §5.5), with the architecture of §6.2.5. One model is trained per
FACS cell line (``K562`` or ``hPSC``) on 512 bp one-hot CRE windows; the
five-way head of the main benchmark is replaced by a single head.

Model wrapped
-------------
``ResNeXt_conv1_128_btnk_2dense.ResNeXt`` (vendored ResNeXt-34, see that
module's docstring). Trained from scratch -- no NvwaCE pretrained weights
are loaded.

Expected on disk
----------------
* ``../_shared/load_facs_data.py`` -- **missing from this checkout**. It
  supplies ``load_facs_for_cell_line(cell_line)`` returning
  ``(trainset, validset, testset)`` dicts with ``'sequence'``
  ``(N, 512, 4)``, ``'label'`` ``(N,)`` and ``'length'``. Labels follow
  §5.5: tRRA > 2 is positive.
* ``ResNeXt_conv1_128_btnk_2dense.py`` in the same directory.

Outputs (``--output_dir``, default ``./FACS_benchmark/NVWACE``)::

    <output_dir>/<cell_line>/NVWACE_FACS.best.pt
    <output_dir>/<cell_line>/NVWACE_FACS_test_predictions.csv
    <output_dir>/<cell_line>/NVWACE_FACS_test_metrics.csv

Entry point
-----------
    python train_FACS_NVWACE.py --cell_line K562 --device cuda:0

Deviations from §5.5 (code, not paper)
-------------------------------------
The paper's FACS protocol uses AdamW with weight decay 0.01, a composite
Smooth-L1 + auxiliary-BCE objective, a boosted sampler with 55% positive
mass and m2a-based model selection. This script uses weight decay 1e-5,
plain ``BCEWithLogitsLoss``, no sampler and selects the checkpoint by
validation AUROC. It is therefore a simplified FACS protocol applied to
the §6.2.5 architecture, not a faithful reproduction of §5.5.
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
from ResNeXt_conv1_128_btnk_2dense import ResNeXt, BasicBlock
# ``load_facs_data`` is expected in ``../_shared``, which does **not** exist
# in this checkout; unlike the eScreen FACS script there is no fallback
# cluster path here, so the import fails unless the helper is supplied.
# ``ResNeXt_conv1_128_btnk_2dense`` is imported as a sibling module, which
# assumes this directory is on ``sys.path`` (true when the script is run
# from ``Analysis/benchmark/NVWACE``); it is not a package import, so
# ``__pycache__``/``-m`` invocations behave differently.
from load_facs_data import load_facs_for_cell_line


class ResNeXt1Head(nn.Module):
    """ResNeXt-34 trunk with a shared feature layer and one output head.

    Same trunk, shared layer and head shape as the five-head benchmark model,
    but with a single head because only one cell line is modelled.

    Parameters
    ----------
    num_heads : int, default 1
        Kept for symmetry with the benchmark wrapper; passing anything other
        than 1 would silently build an unused classifier inside ``backbone``
        while this class still exposes exactly one head.

    Attributes
    ----------
    backbone : ResNeXt
        Trunk only; its NvwaCE bottleneck/classifier are never called.
    bottleneck : DenseLayer
        Bound but unused -- :meth:`forward` skips it.
    shared : nn.Sequential
        ``Linear(1024, 256) -> BatchNorm1d -> GELU -> Dropout(0.2)``.
    head : nn.Sequential
        ``Linear(256, 32) -> BatchNorm1d -> GELU -> Dropout(0.1) ->
        Linear(32, 1)``, the per-cell-line head of §6.2.5.
    """

    def __init__(self, num_heads=1):
        """Build the trunk wrapper, the shared layer and the single head.

        Parameters
        ----------
        num_heads : int, default 1
            Only used to size the unused classifier inside ``backbone``.
        """
        super().__init__()
        self.backbone = ResNeXt(BasicBlock, [3, 4, 6, 3], num_classes=num_heads)
        self.bottleneck = self.backbone.bottleneck
        self.shared = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(256, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 1))

    def forward(self, x):
        """Return one logit per sample.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, 4, 1, L)`` one-hot DNA; ``(B, 4, L)`` is accepted and gets
            the height axis inserted.

        Returns
        -------
        torch.Tensor
            ``(B,)`` raw logits (``flatten()`` removes the trailing 1).
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(2)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x); x = self.backbone.relu(x); x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x); x = self.backbone.layer2(x)
        x = self.backbone.layer3(x); x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.shared(x)
        return self.head(x).flatten()


def prepare_sequence_resnext(data_dict):
    """Convert ``(N, L, 4)`` one-hot DNA to the ``(N, 4, 1, L)`` 2-D layout.

    Parameters
    ----------
    data_dict : dict
        Must contain ``'sequence'`` of shape ``(N, L, 4)``.

    Returns
    -------
    torch.Tensor
        ``float32`` tensor ``(N, 4, 1, L)`` with the A/C/G/T axis at
        position 1 and a singleton height axis at position 2.

    Notes
    -----
    Byte-for-byte duplicate of the helper in ``train_NVWACE.py``; the two
    are kept separate so that each script is runnable on its own.
    """
    seq = data_dict['sequence']
    seq = seq.transpose(0, 2, 1)
    return torch.tensor(seq[:, :, None, :], dtype=torch.float)


def train_nvwace_facs(cell_line, device='cuda:0', epochs=50, batch_size=64,
                        lr=1e-3, weight_decay=1e-5, earlystop=10,
                        output_dir='./FACS_benchmark/NVWACE'):
    """Train NVWACE on the FACS readout of a single cell line.

    Parameters
    ----------
    cell_line : {'K562', 'hPSC'}
        FACS cell line; one independent model per line.
    device : str, default 'cuda:0'
        Torch device string (hard-coded default).
    epochs : int, default 50
        Maximum number of epochs.
    batch_size : int, default 64
        Training batch size; matches §6.2.5's NVWACE setting.
    lr : float, default 1e-3
        AdamW learning rate (matches §5.5).
    weight_decay : float, default 1e-5
        AdamW weight decay. §5.5 reports 0.01 for the FACS experiment, so
        this default does *not* match the paper.
    earlystop : int, default 10
        Epochs without validation-AUROC improvement before stopping (§5.5
        specifies 15 for eScreen; 10 is the value used here for every
        baseline).
    output_dir : str, default './FACS_benchmark/NVWACE'
        Root directory; artefacts go to ``<output_dir>/<cell_line>/``.

    Returns
    -------
    dict
        ``{'cell_line', 'model', 'AUROC', 'AUPRC', 'n_pos', 'n_neg'}`` on the
        test split.

    Notes
    -----
    Validation runs once per epoch (not every ``check_step`` steps), the
    objective is plain BCE-with-logits, and both reported metrics are
    computed on sigmoid probabilities (§6.1, §7.1).
    """
    # 1. Data
    print(f'\n[1/4] Loading FACS data for {cell_line}...')
    trainset, validset, testset = load_facs_for_cell_line(cell_line)

    save_dir = f'{output_dir}/{cell_line}'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'{save_dir}/NVWACE_FACS'

    # 2. Model
    print(f'\n[2/4] Building model...')
    torch.cuda.empty_cache()
    gc.collect()
    model = ResNeXt1Head(num_heads=1).to(device)
    total = sum(p.numel() for p in model.parameters())
    # BatchNorm1d is used in ``shared``/``head``, so a trailing batch of size 1
    # would make the BatchNorm update raise. Batching here is manual (no
    # ``drop_last``), so a split whose size is exactly 1 mod ``batch_size``
    # would fail on its last step; nothing in the script guards against it.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total:,}, Trainable: {trainable:,}')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Default weight decay is 1e-5 here, matching §6.2.5 for the main
    # benchmark. Note §5.5 (FACS) reports 0.01 for this experiment, so the
    # FACS runs do not reproduce the paper's FACS optimizer settings.
    # No LR schedule, no warm-up, and no class-balanced sampler.
    loss_fn = nn.BCEWithLogitsLoss()

    # Pre-compute validation tensors
    val_seq_all = prepare_sequence_resnext(validset).numpy()
    # Validation (and, later, test) sequences are converted once and kept in
    # host memory; batches are copied to the device inside the loop. This keeps
    # the full split off the GPU, which matters at 512 bp x 4 channels.
    # ``val_y`` is read twice below (a harmless leftover of an earlier revision).
    val_y = validset['label']
    val_batch_size = min(64, len(val_y))
    val_y = validset['label']

    # 3. Train
    print(f'\n[3/4] Training ({epochs} epochs)...')
    best_val = -float('inf')
    count = 0

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        indices = np.random.permutation(trainset['length'])

        for i in range(0, trainset['length'], batch_size):
            batch_idx = indices[i:i + batch_size]
            x = prepare_sequence_resnext(
                {'sequence': trainset['sequence'][batch_idx]}).to(device)
            y = torch.tensor(trainset['label'][batch_idx], dtype=torch.float, device=device)

            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Global gradient clipping at norm 1.0, as in §6.2.5.
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
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

    # 4. Evaluate
    print(f'\n[4/4] Evaluating on test set...')
    # The test split is small enough to be scored in a single forward pass,
    # unlike validation; no batching loop is needed.
    test_seq = prepare_sequence_resnext(testset).to(device)
    model.eval()
    with torch.no_grad():
        pt = torch.sigmoid(model(test_seq)).cpu().numpy()

    y_true = testset['label']
    test_auroc = roc_auc_score(y_true, pt)
    prec, rec, _ = precision_recall_curve(y_true, pt)
    test_auprc = auc(rec, prec)

    print(f'\n  === {cell_line} FACS NVWACE ===')
    print(f'  AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}')

    pred_df = pd.DataFrame({'true_label': y_true, 'pred_prob': pt})
    pred_df.to_csv(f'{save_name}_test_predictions.csv', index=False)
    metrics = {'cell_line': cell_line, 'model': 'NVWACE',
                'AUROC': test_auroc, 'AUPRC': test_auprc,
                'n_pos': int(y_true.sum()), 'n_neg': int((1 - y_true).sum())}
    pd.DataFrame([metrics]).to_csv(f'{save_name}_test_metrics.csv', index=False)
    return metrics


def main():
    """Parse CLI arguments, run :func:`train_nvwace_facs`, print the test AUROC."""
    parser = argparse.ArgumentParser(description='NVWACE FACS training')
    parser.add_argument('--cell_line', default='K562', choices=['K562', 'hPSC'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--output_dir', default='./FACS_benchmark/NVWACE')
    args = parser.parse_args()
    metrics = train_nvwace_facs(
        cell_line=args.cell_line, device=args.device,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, earlystop=args.earlystop,
        output_dir=args.output_dir)
    print(f'\nDone! {args.cell_line}: AUROC={metrics["AUROC"]:.4f}')


if __name__ == '__main__':
    main()

