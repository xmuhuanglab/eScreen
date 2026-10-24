#!/usr/bin/env python3
"""
eScreen FACS training — single cell line, single linear head.
Backbone (pretrained PWM) + linear output.
Train/valid/test per cell line, 7:2:1 split.

Read this as the *first-party* arm of the FACS comparison in Supplementary
Methods §5.5 (FACS readout-specific evaluation), not as a baseline
re-implementation. The sequence encoder is the repository's own
``escreen.eSCREEN_backbone`` (Supplementary Methods §2.1) imported as a
library: this file contains no copy of the ``SH2core`` trunk, and
``escreen/escreen.py`` in the package root stays the authoritative
definition of the backbone.

Model wrapped
-------------
``eSCREEN_backbone`` in ``filter_type='pwm'`` mode (the 637 Vierstra motif
PWMs plus their reverse complements, i.e. 1,274 frozen convolution
channels, §2.1) followed by a single ``nn.Linear(256, 1)`` head. There is
no cell-type conditioning: one independent model is trained per FACS cell
line, so a plain linear head replaces the five-way multi-head design used
by the benchmark scripts in the sibling directories.

Expected on disk
----------------
Inputs. Absolute paths below are hard-coded for the authors' Linux
cluster:

* ``BACKBONE_CKPT`` -- pretrained PWM backbone checkpoint
  (``backbone_20260427_pwm_29.pth``), loaded with ``strict=False``.
* ``MEME_PATH`` -- ``consensus_pwms.meme``; the only source of the PWM
  filters, padded to a maximum motif width of 35.
* FACS sequences/labels via ``load_facs_for_cell_line`` from the helper
  module ``load_facs_data``. That module is *not* shipped in this
  checkout; it is resolved from the cluster path pushed onto ``sys.path``.
  Upstream, labels follow §5.5: activity is the tRRA score and a CRE is
  positive when tRRA > 2.

Outputs (under ``--output_dir``, default ``./FACS_benchmark/escreen``)::

    <output_dir>/<cell_line>/escreen_FACS.best.pt
    <output_dir>/<cell_line>/escreen_FACS_test_predictions.csv
    <output_dir>/<cell_line>/escreen_FACS_test_metrics.csv
    <output_dir>/<cell_line>/escreen_FACS_test_summary.txt

Entry point
-----------
    python train_FACS_escreen.py --cell_line K562 --device cuda:0

Deviations from Supplementary Methods §5.5 (code, not paper)
-----------------------------------------------------------
§5.5 describes a composite objective (Smooth L1 on logistic-normalised
RRA plus 0.35x auxiliary BCE with positive class weight 10), a boosted
sampler targeting 55% positive mass, learning rate 1e-3, and checkpoint
selection by the m2a composite score. This script instead trains with
plain ``BCEWithLogitsLoss`` on binarised labels, without a sampler, at a
default learning rate of 3e-4, and selects the checkpoint by validation
AUROC. The numbers it produces therefore come from the simplified
protocol implemented here, not from the protocol written up in §5.5.
"""
import argparse, os, sys, gc, pickle, warnings, ctypes
warnings.filterwarnings('ignore')
# Make CUDA kernels execute synchronously so that an asynchronous failure
# is reported at the offending call instead of surfacing later. Must be
# set before torch is imported (below). This costs throughput and was only
# enabled while debugging.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ── LD_LIBRARY_PATH ──
# LD_LIBRARY_PATH workaround for the authors' cluster image, which carried
# several libstdc++/CUDA builds: the conda environment libraries are
# appended to the search path and libstdc++.so.6 is preloaded with
# RTLD_GLOBAL. All three paths are hard-coded. Note that setting
# LD_LIBRARY_PATH at this point only affects *child* processes -- the
# dynamic loader of the already-running interpreter has cached its search
# paths -- so the RTLD_GLOBAL preload is what actually matters here.
conda_lib = '/cluster2/huanglab/liquan/miniconda3/envs/shin/lib'
torch_lib = conda_lib + '/python3.10/site-packages/torch/lib'
_ld_paths = [p for p in (conda_lib, torch_lib) if p not in os.environ.get('LD_LIBRARY_PATH', '')]
if _ld_paths:
    os.environ['LD_LIBRARY_PATH'] = ':'.join(_ld_paths) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
_stdcpp = os.path.join(conda_lib, 'libstdc++.so.6')
if os.path.exists(_stdcpp):
    try: ctypes.CDLL(_stdcpp, mode=ctypes.RTLD_GLOBAL)
    except: pass

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Import roots. The first entry, ``../_shared``, does **not** exist in this
# checkout: the shared dataset/metric helpers were kept on the cluster, so
# ``load_facs_data`` and ``motif`` come from the absolute paths below. The
# two cluster entries are hard-coded. They also decide which ``escreen`` package wins: with the
# cluster directory first on the path, ``from escreen import ...`` picks
# the cluster copy, not ``D:\GitHub\eScreen\escreen``. When both are
# importable the repository package is the authoritative one.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
sys.path.insert(0, '/cluster2/huanglab/liquan/data/eSCREEN/ANLYSIS/code/')
sys.path.insert(0, '/cluster2/huanglab/liquan/data/eSCREEN/202604_rebuttal/CC/20260523')

from escreen import eSCREEN_backbone
from motif import load_pwm_from_meme_c
from load_facs_data import load_facs_for_cell_line

# ── Paths ──
# Hard-coded absolute paths (cluster-specific).
# ``BACKBONE_CKPT`` holds the distilled/pretrained PWM backbone -- the
# eScreen encoder of §2.1; ``MEME_PATH`` supplies the fixed PWM input layer.
BACKBONE_CKPT = '/cluster2/huanglab/liquan/data/eSCREEN/202604_rebuttal/model_weights/backbone_20260427_pwm_29.pth'
MEME_PATH = "/cluster2/huanglab/liquan/motif/consensus_pwms.meme"


# ═══════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════
class eScreen_FACS(nn.Module):
    """eScreen backbone plus a single scalar output head.

    Parameters
    ----------
    backbone : nn.Module
        An ``escreen.eSCREEN_backbone`` instance already on the target
        device. It maps one-hot DNA ``(B, 512, 4)`` to a pooled embedding
        ``(B, d_model)`` (mean over the 512 positions, §2.1) and carries no
        cell-type information.
    d_model : int, default 256
        Width of the pooled embedding: 256 for the full task-specific
        configuration and 128 for the distilled backbone (§2.1).

    Attributes
    ----------
    head : nn.Linear
        ``d_model -> 1``. It emits a raw logit; the sigmoid is applied by
        the metric code and never inside :meth:`forward`, so that
        ``BCEWithLogitsLoss`` receives logits as it requires.

    Notes
    -----
    The encoder is *not* frozen by :func:`build_model`; the whole backbone
    is fine-tuned on the FACS data (see the ``requires_grad`` loop there).
    :meth:`forward` squeezes the trailing singleton with ``flatten()`` and
    therefore returns ``(B,)``, not ``(B, 1)``.
    """

    def __init__(self, backbone, d_model=256):
        """Wrap a pre-built encoder and attach the scalar head.

        Parameters
        ----------
        backbone : nn.Module
            Encoder returning ``(B, d_model)`` sequence-level embeddings.
        d_model : int, default 256
            Embedding width the backbone is expected to produce.
        """
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        """Score a batch of one-hot CRE windows.

        Parameters
        ----------
        x : torch.Tensor
            One-hot DNA ``(B, 512, 4)``, float32; the encoder casts to
            bfloat16 internally. Base order is A, C, G, T.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B,)``; apply ``sigmoid`` for probabilities.
        """
        emb = self.backbone(x)  # (B, d_model)
        return self.head(emb).flatten()

    @torch.no_grad()
    def predict(self, data, batch_size=256, device='cpu', verbose=True):
        """Run batched inference over a dict-style dataset split.

        Parameters
        ----------
        data : dict
            Split returned by ``load_facs_for_cell_line`` with keys
            ``'sequence'`` (``(N, 512, 4)`` one-hot, NumPy), ``'label'``
            (``(N,)`` in {0, 1}) and ``'length'`` (``N``).
        batch_size : int, default 256
            Number of windows per forward pass.
        device : str, default 'cpu'
            Torch device string; callers pass the training device.
        verbose : bool, default True
            Show a ``tqdm`` progress bar.

        Returns
        -------
        numpy.ndarray
            ``float32`` logits of shape ``(N,)`` in input order; never
            sigmoid-ed here.

        Notes
        -----
        ``torch.no_grad`` keeps the activation graph of the Hyena blocks off
        the GPU, which is what makes a single pass over the whole split fit.
        """
        self.eval()
        preds = []
        indices = range(0, data['length'], batch_size)
        pbar = tqdm(indices) if verbose else indices
        for i in pbar:
            x = torch.tensor(data['sequence'][i:i + batch_size], dtype=torch.float, device=device)
            out = self(x)
            preds.append(out.cpu().numpy())
        return np.concatenate(preds)


def build_model(device='cuda:0'):
    """Build the PWM eScreen encoder, restore its weights, attach the head.

    Parameters
    ----------
    device : str, default 'cuda:0'
        Device onto which both the backbone and the wrapper are moved.

    Returns
    -------
    eScreen_FACS
        The wrapped model, ready for the training loop.

    Notes
    -----
    * The PWMs are materialised as tensors here rather than inside the
      backbone so that the frozen motif weights follow this script's
      dtype/device handling; the backbone concatenates them into the
      1,274-channel input convolution described in §2.1.
    * ``strict=False`` when restoring ``BACKBONE_CKPT`` silently tolerates
      missing or unexpected keys. A checkpoint trained with the 128-wide
      distilled backbone would therefore load partially and leave the rest
      randomly initialised.
    * Despite the module-level phrasing "pretrained backbone + linear
      output", every backbone parameter is set trainable here, i.e. this
      is full fine-tuning of the encoder.
    * ``celltype_num=64`` is only a constructor requirement of
      ``eSCREEN_backbone``; the FACS head ignores cell-type identity.
    """
    d_model = 256
    num_filters = 256

    # Load PWM motifs
# Forward and reverse-complement PWMs, each ``(num_motifs, 4, 35)``.
# ``eSCREEN_backbone`` stacks the two blocks along the output-channel axis
# to obtain the 1,274 frozen motif channels of §2.1. Because the PWMs
# become convolution weights, the ``num_filters`` argument is unused in
# 'pwm' mode.
    motifs_f, motifs_r, _, _ = load_pwm_from_meme_c(MEME_PATH, max_length=35)
    kernel_fwd = torch.tensor(motifs_f, dtype=torch.float)
    kernel_rev = torch.tensor(motifs_r, dtype=torch.float)

    backbone = eSCREEN_backbone(
        filter_type='pwm', kernel_fwd=kernel_fwd, kernel_rev=kernel_rev,
        d_model=d_model, num_filters=num_filters,
        use_flash_attn=False, proj_groups=1,
        seq_length=512, celltype_num=64, lr=1e-5, device=device,
    )
    # Load pretrained backbone weights
# ``strict=False`` is intentional but silent: a shape/name mismatch
# degrades to a partially initialised encoder without raising. Check the
# printed trainable parameter count if downstream metrics look like noise.
    state = torch.load(BACKBONE_CKPT, map_location=device)
    backbone.load_state_dict(state, strict=False)
    print(f'[Model] Loaded backbone weights from {BACKBONE_CKPT}')
    for p in backbone.parameters():
        p.requires_grad = True

    model = eScreen_FACS(backbone=backbone.to(device), d_model=d_model).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[Model] Total params: {total:,}, Trainable: {trainable:,}')
    return model


# ═══════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════
def train_facs(cell_line, device='cuda:0', epochs=50, batch_size=256,
               lr=3e-4, earlystop=15, check_step=500,
               output_dir='./FACS_benchmark/escreen'):
    """Fine-tune eScreen on the FACS readout of a single cell line.

    Parameters
    ----------
    cell_line : {'K562', 'hPSC'}
        FACS cell line to train; each line gets its own model.
    device : str, default 'cuda:0'
        Torch device string (hard-coded default).
    epochs : int, default 50
        Maximum number of passes over the training split.
    batch_size : int, default 256
        Mini-batch size used for training and for split-wide inference.
    lr : float, default 3e-4
        AdamW learning rate. Note that §5.5 reports 1e-3 for this experiment.
    earlystop : int, default 15
        Consecutive epochs without validation-AUROC improvement before
        stopping, matching "15 epochs without improvement" in §5.5.
    check_step : int, default 500
        Accepted for interface parity with the step-based benchmark
        scripts, but unused: validation runs once per epoch here.
    output_dir : str, default './FACS_benchmark/escreen'
        Root directory; artefacts go to ``<output_dir>/<cell_line>/``.

    Returns
    -------
    dict
        ``{'cell_line', 'model', 'AUROC', 'AUPRC', 'n_pos', 'n_neg'}``,
        measured on the held-out test split after restoring the best
        validation checkpoint.

    Notes
    -----
    The objective is plain ``BCEWithLogitsLoss`` on binarised labels, not
    the Smooth-L1 + auxiliary-BCE composite of §5.5, and no class-balanced
    sampler is used. Gradients are clipped to a global norm of 1.0. Both
    AUROC and AUPRC are computed on sigmoid probabilities (§6.1, §7.1).
    """

    # 1. Data
    print(f'\n[1/4] Loading FACS data for {cell_line}...')
    trainset, validset, testset = load_facs_for_cell_line(cell_line)

    save_dir = f'{output_dir}/{cell_line}'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'{save_dir}/escreen_FACS'

    # 2. Model
    print(f'\n[2/4] Building model...')
    torch.cuda.empty_cache()
    gc.collect()
    model = build_model(device=device)

    # 3. Train
    print(f'\n[3/4] Training ({epochs} epochs)...')
# Weight decay 0.01 matches Supplementary Methods §5.5; the learning-rate
# default (3e-4) does not (the manuscript reports 1e-3). No LR schedule or
# warm-up is used in either case.
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -float('inf')
    count = 0
    valid_count = 0

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

# Epoch-level shuffling over sample indices, drawn from the global NumPy
# RNG (never seeded, so runs are not bit-reproducible). The class-balanced
# / label-boosted sampler used for main eScreen training (§2.3.4, §5.5) is
# *not* applied here. Batching is manual, so the last batch of an epoch may
# be shorter than ``batch_size`` (no ``drop_last``).
        indices = np.random.permutation(trainset['length'])
        for i in range(0, trainset['length'], batch_size):
            batch_idx = indices[i:i + batch_size]
            x = torch.tensor(trainset['sequence'][batch_idx], dtype=torch.float).to(device)
            y = torch.tensor(trainset['label'][batch_idx], dtype=torch.float).to(device)

            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
# Global gradient-norm clipping at 1.0: the Hyena blocks are numerically
# delicate in the first epochs, and clipping keeps one bad batch from
# destroying the checkpoint.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            valid_count += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}", end='  ')

        # Validation at end of epoch
        pv = model.predict(validset, batch_size=batch_size, device=device, verbose=False)
        yv = validset['label']
# Guard validation AUC only: a degenerate validation slice (a single class
# present) makes ``roc_auc_score`` raise. The 0.0 fallback keeps the loop
# alive, but note that 0.0 still beats the initial ``best_val = -inf``, so
# the first such epoch always writes a checkpoint.
        try:
            val_auroc = roc_auc_score(yv, torch.sigmoid(torch.tensor(pv)).numpy())
        except:
            val_auroc = 0

        if val_auroc > best_val:
            best_val = val_auroc
            count = 0
            torch.save(model.state_dict(), save_name + '.best.pt')
            print(f"Val AUROC: {val_auroc:.4f}  (saved)  ↑")
        else:
            count += 1
            print(f"Val AUROC: {val_auroc:.4f}  (best: {best_val:.4f})  -")
            if count >= earlystop:
                print(f'  Early stop at epoch {epoch+1}')
                break

    # Load best
    best_path = save_name + '.best.pt'
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f'[Train] Loaded best model from {best_path}')

    # 4. Evaluate
    print(f'\n[4/4] Evaluating on test set...')
    pt = model.predict(testset, batch_size=batch_size, device=device, verbose=True)
    pt_prob = torch.sigmoid(torch.tensor(pt)).numpy()

    y_true = testset['label']
    test_auroc = roc_auc_score(y_true, pt_prob)
# AUPRC as the trapezoidal area under the precision-recall curve rather
# than ``average_precision_score``. The two differ by interpolation scheme
# and are not interchangeable across the benchmark scripts; §7.1 only names
# the metric, so what this script reports is the trapezoid variant.
    prec, rec, _ = precision_recall_curve(y_true, pt_prob)
    test_auprc = auc(rec, prec)

    print(f'\n  === {cell_line} FACS eScreen ===')
    print(f'  AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}')
    print(f'  n_pos={int(y_true.sum())}, n_neg={int((1-y_true).sum())}')

    # Save predictions
    pred_df = pd.DataFrame({
        'true_label': y_true,
        'pred_prob': pt_prob,
        'pred_logit': pt,
    })
    pred_path = f'{save_name}_test_predictions.csv'
    pred_df.to_csv(pred_path, index=False)
    print(f'  Predictions saved to {pred_path}')

    # Save metrics
    metrics = {'cell_line': cell_line, 'model': 'eScreen',
               'AUROC': test_auroc, 'AUPRC': test_auprc,
               'n_pos': int(y_true.sum()), 'n_neg': int((1 - y_true).sum())}
    pd.DataFrame([metrics]).to_csv(f'{save_name}_test_metrics.csv', index=False)
    with open(f'{save_name}_test_summary.txt', 'w') as f:
    # A plain-text twin of the metrics CSV: same two numbers, no header,
    # written for quick `grep`-ability of a single run's result.
        f.write(f'eScreen FACS - {cell_line}\n')
        f.write(f'AUROC={test_auroc:.6f}, AUPRC={test_auprc:.6f}\n')

    print(f'  Metrics saved to {save_name}_test_metrics.csv')
    return metrics


def main():
    """Parse CLI arguments, run :func:`train_facs`, print the test AUROC.

    Returns
    -------
    None
        The metric dict from :func:`train_facs` is only used for the final
        one-line print; every artefact is written inside ``train_facs``.

    Notes
    -----
    ``--cell_line`` is restricted to the two FACS cell lines of §5.5
    (``K562``, ``hPSC``); ``--earlystop`` and ``--output_dir`` are forwarded
    unchanged.
    """
    parser = argparse.ArgumentParser(description='eScreen FACS training')
    parser.add_argument('--cell_line', default='K562', choices=['K562', 'hPSC'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--earlystop', type=int, default=15)
    parser.add_argument('--output_dir', default='./FACS_benchmark/escreen')
    args = parser.parse_args()

    metrics = train_facs(
        cell_line=args.cell_line,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        earlystop=args.earlystop,
        output_dir=args.output_dir,
    )
    print(f'\nDone! {args.cell_line}: AUROC={metrics["AUROC"]:.4f}')


if __name__ == '__main__':
    main()
