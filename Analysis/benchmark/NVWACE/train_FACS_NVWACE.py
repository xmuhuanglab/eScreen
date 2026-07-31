#!/usr/bin/env python3
"""NVWACE (ResNeXt34) FACS training — single cell line, single head."""
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
from load_facs_data import load_facs_for_cell_line


class ResNeXt1Head(nn.Module):
    """ResNeXt34 backbone + single output head."""

    def __init__(self, num_heads=1):
        super().__init__()
        self.backbone = ResNeXt(BasicBlock, [3, 4, 6, 3], num_classes=num_heads)
        self.bottleneck = self.backbone.bottleneck
        self.shared = nn.Sequential(
            nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2))
        self.head = nn.Sequential(
            nn.Linear(256, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 1))

    def forward(self, x):
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
    """Convert (N, L, 4) OHE to (N, 4, 1, L) for ResNeXt."""
    seq = data_dict['sequence']
    seq = seq.transpose(0, 2, 1)
    return torch.tensor(seq[:, :, None, :], dtype=torch.float)


def train_nvwace_facs(cell_line, device='cuda:0', epochs=50, batch_size=64,
                        lr=1e-3, weight_decay=1e-5, earlystop=10,
                        output_dir='./FACS_benchmark/NVWACE'):
    """Train NVWACE on FACS data for one cell line."""
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {total:,}, Trainable: {trainable:,}')

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Pre-compute validation tensors
    val_seq_all = prepare_sequence_resnext(validset).numpy()
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

