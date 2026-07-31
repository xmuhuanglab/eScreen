#!/usr/bin/env python3
"""Malinois (BassetBranched) FACS training — single cell line, single head."""
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
    """BassetBranched backbone + single output head."""

    def __init__(self, input_len=512):
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
        x = self.conv_block(x)
        x = self.shared_fc(x)
        return self.head(x).flatten()


def prepare_sequence_malinois(data_dict):
    """Convert (N, L, 4) OHE to (N, 4, L) for Conv1d."""
    seq = data_dict['sequence']
    return torch.tensor(seq.transpose(0, 2, 1), dtype=torch.float)


def train_malinois_facs(cell_line, device='cuda:0', epochs=50, batch_size=64,
                        lr=1e-3, weight_decay=1e-5, earlystop=10,
                        output_dir='./FACS_benchmark/Malinois'):
    """Train Malinois on FACS data for one cell line."""
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