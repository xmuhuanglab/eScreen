# -*- coding: utf-8 -*-
"""
FACS dataset — compute perturbation scores from HDF5 and evaluate.

Pipeline (from AlphaGenome_perturbation_FACS.ipynb):
  1. Compute mask_score / shuffle_score from HDF5 (sum over the whole 1 Mb window)
  2. Merge with FACS data, create binary labels with RRA > 1
  3. Evaluate AUROC / AUPRC per cell line on the FACS test split
     (MASK / SHUFFLE strategies)
  4. Threshold sensitivity analysis + Perturbation Score vs RRA correlation
     + quintile plot + CAGE tracks

Usage:
    python analyze_facs_scores.py
"""
import json
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from alphagenome_common import (
    m2a_combo_score, per_cell_line_metrics, print_metrics,
    PERTURB_CENTER,
)


OUTPUT_H5 = 'data/AlphaGenome_FACS_perturbation.h5'
SCORE_CSV = 'data/AlphaGenome_FACS_perturbation_scores.csv'
CHECKPOINT_FILE = 'data/checkpoint_FACS_perturb.pkl'


def compute_scores_from_h5(cre_df=None):
    """Compute mask_score / shuffle_score for every sample in HDF5."""
    with h5py.File(OUTPUT_H5, 'r') as f:
        cage = f['cage']
        sample_ids = list(cage.keys())
        print(f'Total samples in HDF5: {len(sample_ids)}')

        records = []
        for sid in tqdm(sample_ids, desc='Scoring'):
            grp = cage[sid]
            wt      = grp['wt'][:].astype(np.float32)
            mask    = grp['mask'][:].astype(np.float32)
            shuffle = grp['shuffle'][:].astype(np.float32)

            mask_score    = np.sum(wt - mask)
            shuffle_score = np.sum(wt - shuffle)

            records.append({
                'sample_id': sid,
                'cell_line': grp.attrs['cell_line'],
                'mask_score': mask_score,
                'shuffle_score': shuffle_score,
            })

    score_df = pd.DataFrame(records)
    print(f'Scored {len(score_df)} samples')
    return score_df


def load_and_merge(score_df, cre_csv='data/FACS_fluorescence_dataset.txt'):
    """Merge scores with the original FACS data, create RRA > 1 binary labels."""
    cre_df = pd.read_csv(cre_csv, sep='\t')

    # Parse sample_id -> id + cell_line
    score_df[['id', 'cl']] = score_df['sample_id'].str.split('@', expand=True)

    merged = pd.merge(score_df, cre_df, left_on=['id', 'cell_line'],
                      right_on=['id', 'cell_line'], how='left')
    merged = merged.dropna(subset=['RRA'])

    # Binary label: RRA > 1
    merged['label'] = (merged['RRA'] > 1.0).astype(int)

    print(f'Merged: {len(merged)} samples')
    print('Label distribution:')
    print(merged.groupby('cell_line')['label'].value_counts().unstack(fill_value=0))

    # Save full results
    merged.to_csv(SCORE_CSV, sep='\t', index=False, float_format='%.6f')
    print(f'\nSaved to {SCORE_CSV}')
    return merged


def load_test_ids():
    """Load FACS test split CRE ids (same as 20260611/run_FACS_benchmark.py)."""
    with open('data/FACS_split_hPSC_ids.json', 'r') as file:
        hPSC_testset = pd.DataFrame(json.load(file)['test'])
        hPSC_testset['cell_line'] = 'hPSC'

    with open('data/FACS_split_K562_ids.json', 'r') as file:
        K562_testset = pd.DataFrame(json.load(file)['test'])
        K562_testset['cell_line'] = 'K562'

    idset = (list(hPSC_testset['cre_ids'] + '@hPSC')
             + list(K562_testset['cre_ids'] + '@K562'))
    return idset


def evaluate_on_test(merged):
    """Evaluate MASK / SHUFFLE strategies on the FACS test split."""
    idset = load_test_ids()
    test = merged[merged['sample_id'].isin(idset)]
    groups = test['cell_line'].values
    y_true = test['label'].values
    print(f'Test samples: {len(groups)}, groups: {dict(pd.Series(groups).value_counts())}')

    # --- MASK strategy ---
    combo_mask, results_mask = m2a_combo_score(test['mask_score'].values, y_true, groups)
    print_metrics(results_mask, 'MASK strategy')
    print(f'm2a_combo = {combo_mask:.4f}')

    # --- SHUFFLE strategy ---
    combo_shuffle, results_shuffle = m2a_combo_score(test['shuffle_score'].values, y_true, groups)
    print_metrics(results_shuffle, 'SHUFFLE strategy')
    print(f'm2a_combo = {combo_shuffle:.4f}')

    return {'combo_mask': combo_mask, 'combo_shuffle': combo_shuffle,
            'results_mask': results_mask, 'results_shuffle': results_shuffle}


def threshold_sensitivity(merged):
    """AUROC / AUPRC changes across different RRA thresholds."""
    idset = load_test_ids()
    test = merged[merged['sample_id'].isin(idset)].copy()
    groups = test['cell_line'].values

    thresholds = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for thr in thresholds:
        y = (test['RRA'] > thr).astype(int)
        n_pos = y.sum()
        if n_pos < 10 or n_pos > len(y) - 10:
            continue
        combo_mask, rm = m2a_combo_score(test['mask_score'].values, y, groups)
        combo_shuf, rs = m2a_combo_score(test['shuffle_score'].values, y, groups)
        rows.append({'threshold': thr, 'n_pos': n_pos, 'n_neg': len(y) - n_pos,
                     'mask_combo': combo_mask, 'shuffle_combo': combo_shuf,
                     'mask_AUROC': np.median([r['AUROC'] for r in rm]),
                     'shuffle_AUROC': np.median([r['AUROC'] for r in rs])})

    thr_df = pd.DataFrame(rows)
    print('\n=== Threshold Sensitivity ===')
    print(thr_df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thr_df['threshold'], thr_df['mask_AUROC'], 'o-', label='MASK AUROC')
    ax.plot(thr_df['threshold'], thr_df['shuffle_AUROC'], 's--', label='SHUFFLE AUROC')
    ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('RRA Threshold')
    ax.set_ylabel('Median AUROC')
    ax.set_title('Threshold Sensitivity')
    ax.legend()
    ax.set_xticks(thresholds)
    plt.tight_layout()
    plt.savefig('data/FACS_threshold_sensitivity.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return thr_df


def correlation_with_rra(merged):
    """Spearman correlation between perturbation score and RRA + scatter plots."""
    print('\n=== Spearman Correlation: Perturbation Score vs RRA ===')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cell, color in [(axes[0], 'K562', '#E74C3C'), (axes[1], 'hPSC', '#3498DB')]:
        sub = merged[merged['cell_line'] == cell]
        r_m, p_m = spearmanr(sub['mask_score'], sub['RRA'])
        r_s, p_s = spearmanr(sub['shuffle_score'], sub['RRA'])
        print(f'{cell} (n={len(sub)}):')
        print(f'  MASK:    rho = {r_m:.4f}, p = {p_m:.2e}')
        print(f'  SHUFFLE: rho = {r_s:.4f}, p = {p_s:.2e}')

        ax.scatter(sub['RRA'], sub['mask_score'], alpha=0.3, s=3, color=color, label='MASK')
        ax.scatter(sub['RRA'], sub['shuffle_score'], alpha=0.15, s=3, color='gray', label='SHUFFLE')
        ax.set_xlabel('RRA Score')
        ax.set_ylabel('Perturbation Score')
        ax.set_title(f'{cell} (n={len(sub)})')
        ax.legend(markerscale=5)
        ax.axhline(0, color='gray', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('data/FACS_perturbation_vs_RRA.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def quintile_plot(merged):
    """Mean MASK perturbation score by RRA quintile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, cell, color in [(axes[0], 'K562', '#E74C3C'), (axes[1], 'hPSC', '#3498DB')]:
        sub = merged[merged['cell_line'] == cell].copy()
        sub['bin'] = pd.qcut(sub['RRA'].rank(method='first'), 5,
                             labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        stats = sub.groupby('bin')['mask_score'].agg(['mean', 'sem'])
        ax.errorbar(range(5), stats['mean'], yerr=stats['sem'],
                    marker='o', capsize=5, color=color, linewidth=2)
        ax.set_xticks(range(5))
        ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        ax.set_xlabel('RRA Quintile')
        ax.set_ylabel('Mean MASK Perturbation Score')
        ax.set_title(cell)
        ax.axhline(0, color='gray', ls='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('data/FACS_perturbation_by_RRA_quintile.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_tracks(merged, sample_id, window_bp=20000, save_prefix='data/FACS_track'):
    """Example: WT / MASK / SHUFFLE CAGE tracks for a sample."""
    with h5py.File(OUTPUT_H5, 'r') as f:
        grp = f['cage'][sample_id]
        wt_data      = grp['wt'][:]
        mask_data    = grp['mask'][:]
        shuffle_data = grp['shuffle'][:]
        chrm = grp.attrs['chr']
        begin = grp.attrs['start']

    half = window_bp // 2
    center = PERTURB_CENTER
    s = center - half
    e = center + half

    x = np.arange(begin + s, begin + e)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, wt_data[s:e],      label='WT',      color='#2C3E50', lw=0.8)
    ax.plot(x, mask_data[s:e],    label='MASK',    color='#E74C3C', lw=0.8, alpha=0.7)
    ax.plot(x, shuffle_data[s:e], label='SHUFFLE', color='#3498DB', lw=0.6, alpha=0.5, ls='--')

    # Annotate the CRE region (original coordinates)
    row = merged[merged['sample_id'] == sample_id].iloc[0]
    ax.axvspan(row['start'], row['end'], color='red', alpha=0.08, label='CRE')
    ax.set_xlabel(f'Position ({chrm})')
    ax.set_ylabel('CAGE signal')
    ax.set_title(f'{sample_id} | RRA={row["RRA"]:.2f} | mask_score={row["mask_score"]:.1f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{sample_id.replace("@", "_")}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    score_df = compute_scores_from_h5()
    merged = load_and_merge(score_df)
    evaluate_on_test(merged)
    threshold_sensitivity(merged)
    correlation_with_rra(merged)
    quintile_plot(merged)

    # Example: top 2 K562 CREs by mask_score
    top_k = merged[merged['cell_line'] == 'K562'].nlargest(2, 'mask_score')
    for sid in top_k['sample_id'].values:
        plot_tracks(merged, sid)

    print('\nOutput files:')
    print(f'  HDF5: {OUTPUT_H5}')
    print(f'  CSV:  {SCORE_CSV}')
    print(f'  PNG:  data/FACS_perturbation_vs_RRA.png')
    print(f'  PNG:  data/FACS_perturbation_by_RRA_quintile.png')
    print(f'  PNG:  data/FACS_threshold_sensitivity.png')


if __name__ == '__main__':
    main()
