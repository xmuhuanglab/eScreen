#!/usr/bin/env python3
"""
HyenaDNA — 5-head finetuning for 5 cell lines, 512bp token input.

Usage:
    python train_HyenaDNA.py --device cuda:0 --epochs 50 --batch_size 32
    python train_HyenaDNA.py --device cuda:1 --model tiny-1k-d256 --freeze_backbone
"""
import argparse, os, sys, gc, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Safe globals for omegaconf (needed for loading HyenaDNA checkpoint)
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.base import ContainerMetadata
torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])

from HyenaDNA.huggingface import HyenaDNAModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from data_utils import (
    build_datasets_from_original,
    per_cell_line_metrics, print_metrics_table, save_metrics,
    P0_CELL_LINES, CELL_LINE_TO_IDX
)


# ── Constants ───────────────────────────────────────────────────────
MAXLEN = {
    'tiny-1k-d128': 1026,
    'tiny-1k-d256': 1026,
    'tiny-16k': 16386,
    'small-32k': 32770,
    'middle-160k': 160002,
    'middle-450k': 450002,
    'large-1m': 1000002,
}


# ── Model ───────────────────────────────────────────────────────────
class HyenaDNA5Head(nn.Module):
    """
    HyenaDNA backbone with 5 output heads (one per cell line).
    Optionally supports cell-type conditioning (notebook-style) via --use_cell_emb.
    """

    def __init__(self, backbone, d_model, cell_type_num=16, num_heads=5,
                 freeze_backbone=True, use_cell_emb=False):
        """Build the shared trunk and the five per-cell-line heads.

        Parameters
        ----------
        backbone : HyenaDNAModel
            Pretrained encoder.
        d_model : int
            Hidden width of the backbone.
        cell_type_num : int, default 16
            Size of the cell-type embedding table.
        num_heads : int, default 5
            One head per benchmark cell line (K562, HepG2, HT29, hPSC, A375).
        freeze_backbone : bool, default True
            Whether to hold the encoder fixed and train only the head stack.
        use_cell_emb : bool, default False
            Concatenate a learned cell-type embedding to the sequence embedding
            instead of relying on the per-cell-line heads alone.
        """
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.use_cell_emb = use_cell_emb
        self.freeze_backbone = freeze_backbone

        if use_cell_emb:
            # Notebook-style: cell type embedding + 5 heads
            self.ct_emb = nn.Embedding(num_embeddings=cell_type_num, embedding_dim=d_model)
            input_dim = d_model * 2
        else:
            input_dim = d_model

        # Shared classifier
        self.shared = nn.Sequential(
            nn.Linear(input_dim, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        ).to(torch.bfloat16)

        # 5 cell-line-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(d_model, 1).to(torch.bfloat16) for _ in range(num_heads)
        ])

        self._apply_freezing()

    def _apply_freezing(self):
        """Freeze or unfreeze the encoder according to ``freeze_backbone``.

        Batch-norm statistics are disabled and the modules pinned to eval mode while
        frozen, so a frozen encoder does not keep shifting its running means.
        """
        for p in self.backbone.parameters():
            p.requires_grad = not self.freeze_backbone
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = not self.freeze_backbone
                if self.freeze_backbone:
                    m.eval()

    def forward(self, x, ct=None):
        # x: (B, L) token IDs
        # HyenaDNA returns (B, L, d_model) hidden states
        """Score a batch for every cell line at once.

        Parameters
        ----------
        x : torch.Tensor
            Token ids, ``[B, L]``.
        ct : torch.Tensor, optional
            Cell-type indices, needed only when ``use_cell_emb`` is set.

        Returns
        -------
        torch.Tensor
            Logits, ``[B, num_heads]``, one column per cell line.

        Notes
        -----
        The sequence summary is the hidden state of the final position, which for
        HyenaDNA carries the aggregated context of the whole window.  Because every
        head is evaluated on every sample, the caller selects the column matching
        each sample's own cell line; one forward pass therefore yields a prediction
        for all five labels without re-running the backbone.
        """
        hidden = self.backbone(x)  # (B, L, d_model)
        # Use last token's embedding ([SEP]/[EOS] token)
        seq_emb = hidden[:, -1, :]  # (B, d_model)

        if self.use_cell_emb and ct is not None:
            ct_emb = self.ct_emb(ct)
            emb = torch.cat([seq_emb, ct_emb], dim=1)
        else:
            emb = seq_emb

        emb = self.shared(emb)
        logits = torch.cat([head(emb) for head in self.heads], dim=1)
        return logits


# ── Build HyenaDNA ──────────────────────────────────────────────────
def build_hyenadna(model_name='tiny-1k-d256', device='cpu'):
    """Load pretrained HyenaDNA model."""
    model_path = f'/path/to/eScreen/code/HyenaDNA/pretrained_model/{model_name}/'

    print(f'  Loading HyenaDNA config from {model_path}config.json')
    with open(model_path + 'config.json', 'r') as f:
        config = json.load(f)

    print(f'  Building HyenaDNA model ({model_name})...')
    backbone = HyenaDNAModel(**config, use_head=False, n_classes=2)

    print(f'  Loading pretrained weights...')
    raw_weight = torch.load(
        model_path + 'weights.ckpt',
        map_location=torch.device('cpu'),
        weights_only=False
    )['state_dict']

    # Remove 'model.' prefix from keys
    weight = {}
    for k, v in raw_weight.items():
        if k.startswith('model.'):
            weight[k[6:]] = v
        else:
            weight[k] = v
    backbone.load_state_dict(weight, strict=False)
    backbone = backbone.to(device)

    # Get d_model from config
    d_model = config.get('d_model', 256)

    return backbone, d_model


# ── Data ────────────────────────────────────────────────────────────
def prepare_tokens(data_dict):
    """Convert token IDs: 0→11(pad), 1→7(A), 2→8(C), 3→9(G), 4→10(T)."""
    tokens = data_dict['token']  # (N, L)
    return torch.tensor(np.where(tokens < 1, 11, tokens + 6), dtype=torch.long)


class HyenaDataset(torch.utils.data.Dataset):
    """Dataset pairing token ids with their binary labels.

    Each item carries the token sequence, the label, and the index of the cell
    line the sample came from.  That index is what selects the matching head
    when the loss is computed.  With cell-type conditioning enabled the
    cell-type id is returned as well.
    """
    def __init__(self, data_dict, use_cell_emb=False):
        """Wrap a pre-tokenised split.

        Parameters
        ----------
        data_dict : dict
            Must contain ``'token'``, ``'label'`` and ``'cell_line_idx'``; also
            ``'cell_type'`` when ``use_cell_emb`` is true.
        use_cell_emb : bool, default False
            Whether to also yield the cell-type id.
        """
        self.seq = prepare_tokens(data_dict)
        self.labels = torch.tensor(data_dict['label'], dtype=torch.float)
        self.cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long)
        self.use_cell_emb = use_cell_emb
        if use_cell_emb:
            self.ct = torch.tensor(data_dict['cell_type'], dtype=torch.long)

    def __len__(self):
        """Number of samples in the split.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """Return one sample.

        Returns
        -------
        tuple
            ``(tokens, label, cell_line_idx)``, or
            ``(tokens, label, cell_line_idx, cell_type)`` when cell-type
            conditioning is enabled.
        """
        if self.use_cell_emb:
            return self.seq[idx], self.labels[idx], self.cl_idx[idx], self.ct[idx]
        return self.seq[idx], self.labels[idx], self.cl_idx[idx]


# ── Training ────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, data_dict, batch_size, device, use_cell_emb=False):
    """Run inference over a whole split and collect per-sample predictions.

    The head matching each sample's own cell line is selected, so the returned
    vector is directly comparable to the labels.

    Parameters
    ----------
    model : HyenaDNA5Head
        Trained model.
    data_dict : dict
        Split to score.
    batch_size : int
        Inference batch size.
    device : torch.device
        Device to run on.
    use_cell_emb : bool, default False
        Whether the model expects a cell-type input.

    Returns
    -------
    numpy.ndarray
        Scores, ``[N]``.
    """
    model.eval()
    all_preds = []
    seq = prepare_tokens(data_dict).to(device)
    cl_idx = torch.tensor(data_dict['cell_line_idx'], dtype=torch.long, device=device)
    ct = torch.tensor(data_dict['cell_type'], dtype=torch.long, device=device) if use_cell_emb else None

    for i in range(0, len(seq), batch_size):
        x_batch = seq[i:i + batch_size]
        cl_batch = cl_idx[i:i + batch_size]
        ct_batch = ct[i:i + batch_size] if ct is not None else None

        if use_cell_emb:
            outputs = model(x_batch, ct=ct_batch)
        else:
            outputs = model(x_batch)

        preds = outputs[torch.arange(len(cl_batch)), cl_batch].float()
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


# ── Main ────────────────────────────────────────────────────────────
def main():
    """Command-line entry point for the five-head benchmark.

    Runs the full pipeline: build the locus-aware splits, train with early
    stopping on the validation set, then report test metrics.  This is the
    HyenaDNA baseline of Supplementary Methods section 6.2.3.

    Defaults are ``epochs=50``, ``batch_size=32``, ``lr=5e-3``,
    ``weight_decay=0.0``, ``earlystop=10`` and ``check_step=2000``.  The
    backbone is frozen by default, so only the heads are trained - the
    fine-tuning regime used for the comparison.  Checkpoints are written under
    ``--output_dir`` (default ``./Models``).
    """
    parser = argparse.ArgumentParser(description='HyenaDNA — 5-head finetuning')
    parser.add_argument('--device', default='cuda:0', help='GPU device, e.g. cuda:0')
    parser.add_argument('--model', default='tiny-1k-d256',
                        choices=list(MAXLEN.keys()), help='HyenaDNA model variant')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--check_step', type=int, default=2000)
    parser.add_argument('--output_dir', default='./Models')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze HyenaDNA backbone (default: True)')
    parser.add_argument('--use_cell_emb', action='store_true', default=False,
                        help='Use cell-type embedding (notebook-style) instead of 5 heads')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f'[HyenaDNA] Device: {device}, Model: {args.model}')
    print(f'[HyenaDNA] Cell lines: {P0_CELL_LINES}')
    print(f'[HyenaDNA] Freeze backbone: {args.freeze_backbone}')
    print(f'[HyenaDNA] Use cell embedding: {args.use_cell_emb}')
    print(f'[HyenaDNA] Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}')

    # 1. Data
    print('\n[1/4] Loading data...')
    trainset, validset, testset, locusset = build_datasets_from_original()

    train_loader = torch.utils.data.DataLoader(
        HyenaDataset(trainset, use_cell_emb=args.use_cell_emb),
        batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # Check that token data exists
    if trainset.get('token') is None:
        print('ERROR: Token data not found in the dataset. HyenaDNA requires token sequences.')
        sys.exit(1)

    # 2. Model
    print('\n[2/4] Building HyenaDNA model...')
    torch.cuda.empty_cache()
    gc.collect()

    # ── Import ──
    from HyenaDNA.huggingface import HyenaFilter

    backbone, d_model = build_hyenadna(model_name=args.model, device=device)
    backbone = backbone.to(torch.bfloat16)  # HyenaDNA expects bfloat16

    model = HyenaDNA5Head(
        backbone=backbone,
        d_model=d_model,
        cell_type_num=16,
        num_heads=5,
        freeze_backbone=args.freeze_backbone,
        use_cell_emb=args.use_cell_emb,
    ).to(device)

    # Set fft_conv_device on all HyenaFilter modules (GPU for tiny model)
    n_filter = 0
    for m in model.modules():
        if isinstance(m, HyenaFilter):
            m.fft_conv_device = device
            n_filter += 1
    if n_filter > 0:
        print(f'  FFT conv device set to {device} on {n_filter} HyenaFilter modules')

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total_params:,}, Trainable: {trainable:,}')

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # 3. Train
    print('\n[3/4] Training...')
    save_name = f'{args.output_dir}/HyenaDNA_5head'
    best_val_score = -float('inf')
    count = 0
    valid_count = 0

    for epoch in range(args.epochs):
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            if args.use_cell_emb:
                x, labels, cl_idx, ct = [b.to(device) for b in batch]
            else:
                x, labels, cl_idx = [b.to(device) for b in batch]
                ct = None

            model.train()
            optimizer.zero_grad()

            if args.use_cell_emb:
                outputs = model(x, ct=ct)
            else:
                outputs = model(x)

            loss = loss_fn(outputs[torch.arange(len(cl_idx)), cl_idx].float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            valid_count += 1

            avg_loss = epoch_loss / n_batches
            print(f"Epoch [{epoch+1}/{args.epochs}] Step {n_batches}, Loss: {avg_loss:.4f}", end='\r', flush=True)

            if valid_count >= args.check_step:
                pv = predict(model, validset, args.batch_size, device,
                             use_cell_emb=args.use_cell_emb)
                results = per_cell_line_metrics(1 / (1 + np.exp(-pv)), validset)
                valid_results = [r for r in results if not np.isnan(r['AUROC'])]
                val_score = np.mean([r['AUROC'] for r in valid_results]) if valid_results else 0

                if val_score > best_val_score:
                    best_val_score = val_score
                    count = 0
                    torch.save(model.state_dict(), save_name + '.best.pt')
                    print(f"\n  Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val mean AUROC: {val_score:.4f}  ↑", flush=True)
                else:
                    count += 1
                    print(f"\n  Epoch [{epoch+1}], Loss: {avg_loss:.4f}, Val mean AUROC: {val_score:.4f}  -", flush=True)
                    if count >= args.earlystop:
                        print(f'  Early stop at epoch {epoch+1}, best val AUROC: {best_val_score:.4f}')
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
    pt = predict(model, testset, args.batch_size, device, use_cell_emb=args.use_cell_emb)
    pt_sigmoid = 1 / (1 + np.exp(-pt))

    results = per_cell_line_metrics(pt_sigmoid, testset)
    print_metrics_table(results, 'Test Set')

    extra = f'HyenaDNA ({args.model}) 5-head\n'
    extra += f'Freeze backbone: {args.freeze_backbone}, Cell emb: {args.use_cell_emb}\n'
    extra += f'Split: original randomCRE.4.pkl → id-level 7:2:1 + P0 filter\n'
    extra += f'Cell lines: {P0_CELL_LINES}\nEpochs: {args.epochs}, LR: {args.lr}\n'
    save_metrics(results, save_name, extra)

    if locusset is not None and locusset['length'] > 0:
        pl = predict(model, locusset, args.batch_size, device, use_cell_emb=args.use_cell_emb)
        pl_sigmoid = 1 / (1 + np.exp(-pl))
        locus_results = per_cell_line_metrics(pl_sigmoid, locusset)
        print_metrics_table(locus_results, 'Locus Holdout (chr11)')

    print('\nDone!')


if __name__ == '__main__':
    main()
