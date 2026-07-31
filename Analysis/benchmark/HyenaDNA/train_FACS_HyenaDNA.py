#!/usr/bin/env python3
"""HyenaDNA FACS training — single cell line, single head, pretrained backbone."""
import argparse, os, sys, gc, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Safe globals for HyenaDNA checkpoint
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.base import ContainerMetadata
torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))
sys.path.insert(0, '/cluster2/huanglab/liquan/data/eSCREEN/ANLYSIS/code/HyenaDNA/')
sys.path.insert(0, '/cluster2/huanglab/liquan/data/eSCREEN/ANLYSIS/code/')
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import HyenaFilter
from load_facs_data import load_facs_for_cell_line


MAXLEN = {
    'tiny-1k-d128': 1026, 'tiny-1k-d256': 1026, 'tiny-16k': 16386,
    'small-32k': 32770, 'middle-160k': 160002, 'middle-450k': 450002, 'large-1m': 1000002,
}


class HyenaDNA1Head(nn.Module):
    """HyenaDNA backbone + single output head."""

    def __init__(self, backbone, d_model, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.freeze_backbone = freeze_backbone

        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.LayerNorm(d_model * 4), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(d_model * 4, d_model), nn.LayerNorm(d_model), nn.GELU(),
)

        self.head = nn.Linear(d_model, 1)
        self._apply_freezing()

    def _apply_freezing(self):
        for p in self.backbone.parameters():
            p.requires_grad = not self.freeze_backbone
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.track_running_stats = not self.freeze_backbone
                if self.freeze_backbone:
                    m.eval()

    def forward(self, x):
        hidden = self.backbone(x)
        emb = hidden[:, -1, :]
        emb = self.shared(emb)
        return self.head(emb).flatten()


def build_hyenadna(model_name='tiny-1k-d256', device='cpu'):
    model_path = f'/cluster2/huanglab/liquan/data/eSCREEN/ANLYSIS/code/HyenaDNA/pretrained_model/{model_name}/'
    with open(model_path + 'config.json', 'r') as f:
        config = json.load(f)
    backbone = HyenaDNAModel(**config, use_head=False, n_classes=2)
    raw_weight = torch.load(
        model_path + 'weights.ckpt', map_location=torch.device('cpu'), weights_only=False)['state_dict']
    weight = {k[6:] if k.startswith('model.') else k: v for k, v in raw_weight.items()}
    backbone.load_state_dict(weight, strict=False)
    backbone = backbone.to(device)
    d_model = config.get('d_model', 256)
    return backbone, d_model


def prepare_tokens(data_dict):
    """Convert token IDs: 0→11(pad), 1→7(A), 2→8(C), 3→9(G), 4→10(T)."""
    tokens = data_dict['token']
    return torch.tensor(np.where(tokens < 1, 11, tokens + 6), dtype=torch.long)


def train_hyenadna_facs(cell_line, device='cuda:0', model_name='tiny-1k-d256',
                        epochs=50, batch_size=16, lr=5e-3, weight_decay=0.0,
                        earlystop=10, freeze_backbone=True,
                        output_dir='./FACS_benchmark/HyenaDNA'):
    """Train HyenaDNA on FACS data for one cell line."""
    print(f'\n[1/4] Loading FACS data for {cell_line}...')
    trainset, validset, testset = load_facs_for_cell_line(cell_line)

    if trainset.get('token') is None:
        print('ERROR: Token data not found. HyenaDNA requires token sequences.')
        return None

    save_dir = f'{output_dir}/{cell_line}'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f'{save_dir}/HyenaDNA_FACS'

    # 2. Model
    print(f'\n[2/4] Building HyenaDNA model ({model_name})...')
    torch.cuda.empty_cache()
    gc.collect()

    backbone, d_model = build_hyenadna(model_name=model_name, device=device)
    backbone = backbone.to(device)

    model = HyenaDNA1Head(
        backbone=backbone, d_model=d_model, freeze_backbone=freeze_backbone,
    ).to(device)

    # Set FFT conv device
    for m in model.modules():
        if isinstance(m, HyenaFilter):
            m.fft_conv_device = device

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total:,}, Trainable: {trainable:,}')

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Pre-compute validation tensors
    val_seq_all = prepare_tokens(validset).numpy()
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
            x = prepare_tokens(
                {'token': trainset['token'][batch_idx]}).to(device)
            y = torch.tensor(trainset['label'][batch_idx], dtype=torch.float, device=device)

            model.train()
            optimizer.zero_grad()
            logits = model(x).float()
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
                vx = torch.tensor(val_seq_all[vi:vi+val_batch_size], dtype=torch.long, device=device)
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
    test_seq_all = prepare_tokens(testset).numpy()
    model.eval()
    all_pt = []
    with torch.no_grad():
        for ti in range(0, len(test_seq_all), val_batch_size):
            tx = torch.tensor(test_seq_all[ti:ti+val_batch_size], dtype=torch.long, device=device)
            all_pt.append(torch.sigmoid(model(tx)).float().cpu().numpy())
    pt = np.concatenate(all_pt)

    y_true = testset['label']
    test_auroc = roc_auc_score(y_true, pt)
    prec, rec, _ = precision_recall_curve(y_true, pt)
    test_auprc = auc(rec, prec)

    print(f'\n  === {cell_line} FACS HyenaDNA ===')
    print(f'  AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}')

    pred_df = pd.DataFrame({'true_label': y_true, 'pred_prob': pt})
    pred_df.to_csv(f'{save_name}_test_predictions.csv', index=False)
    metrics = {'cell_line': cell_line, 'model': 'HyenaDNA',
                'AUROC': test_auroc, 'AUPRC': test_auprc,
                'n_pos': int(y_true.sum()), 'n_neg': int((1 - y_true).sum())}
    pd.DataFrame([metrics]).to_csv(f'{save_name}_test_metrics.csv', index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser(description='HyenaDNA FACS training')
    parser.add_argument('--cell_line', default='K562', choices=['K562', 'hPSC'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model', default='tiny-1k-d256', choices=list(MAXLEN.keys()))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--earlystop', type=int, default=10)
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--output_dir', default='./FACS_benchmark/HyenaDNA')
    args = parser.parse_args()
    metrics = train_hyenadna_facs(
        cell_line=args.cell_line, device=args.device, model_name=args.model,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, earlystop=args.earlystop,
        freeze_backbone=args.freeze_backbone,
        output_dir=args.output_dir)
    print(f'\nDone! {args.cell_line}: AUROC={metrics["AUROC"]:.4f}')


if __name__ == '__main__':
    main()
