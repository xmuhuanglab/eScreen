#!/usr/bin/env python3
"""
Sei FACS training — single cell line, single output head, 4096bp input.
Sei backbone (BSplineConv1D + dilated convs) + single head.
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
    """Sei backbone + single output head."""

    def __init__(self, sequence_length=4096):
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
    val_seq_all = pad_to_4096(validset['sequence']).transpose(0, 2, 1).astype(np.float32)
    val_y = validset['label']
    val_batch_size = min(32, len(val_y))

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        indices = np.random.permutation(trainset['length'])
        for i in range(0, trainset['length'], batch_size):
            batch_idx = indices[i:i + batch_size]
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
