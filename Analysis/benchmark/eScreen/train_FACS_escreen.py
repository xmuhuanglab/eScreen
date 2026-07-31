#!/usr/bin/env python3
"""
eScreen FACS training — single cell line, single linear head.
Backbone (pretrained PWM) + linear output.
Train/valid/test per cell line, 7:2:1 split.
"""
import argparse, os, sys, gc, pickle, warnings, ctypes
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ── LD_LIBRARY_PATH ──
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
sys.path.insert(0, '/cluster2/huanglab/liquan/data/eSCREEN/ANLYSIS/code/')
sys.path.insert(0, '/cluster2/huanglab/liquan/data/eSCREEN/202604_rebuttal/CC/20260523')

from escreen import eSCREEN_backbone
from motif import load_pwm_from_meme_c
from load_facs_data import load_facs_for_cell_line

# ── Paths ──
BACKBONE_CKPT = '/cluster2/huanglab/liquan/data/eSCREEN/202604_rebuttal/model_weights/backbone_20260427_pwm_29.pth'
MEME_PATH = "/cluster2/huanglab/liquan/motif/consensus_pwms.meme"


# ═══════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════
class eScreen_FACS(nn.Module):
    """eScreen backbone + single linear head (no cell conditioning)."""

    def __init__(self, backbone, d_model=256):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        emb = self.backbone(x)  # (B, d_model)
        return self.head(emb).flatten()

    @torch.no_grad()
    def predict(self, data, batch_size=256, device='cpu', verbose=True):
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
    """Build eScreen backbone + simple linear head."""
    d_model = 256
    num_filters = 256

    # Load PWM motifs
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
    """Train eScreen on FACS data for one cell line."""

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
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = -float('inf')
    count = 0
    valid_count = 0

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

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
        f.write(f'eScreen FACS - {cell_line}\n')
        f.write(f'AUROC={test_auroc:.6f}, AUPRC={test_auprc:.6f}\n')

    print(f'  Metrics saved to {save_name}_test_metrics.csv')
    return metrics


def main():
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
