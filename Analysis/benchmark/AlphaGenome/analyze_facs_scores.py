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

What this script implements
---------------------------
The AlphaGenome arm of the FACS readout-specific evaluation in Supplementary
Methods §5.5 (K562 and hPSC), scored with AUROC/AUPRC as specified in §7.1. The
prediction tensors it reads were produced by ``run_facs_perturbation.py``; this
script never calls the model.

Scoring definition
------------------
For each CRE both perturbation scores are accumulated over *all* output bins of
the 1 Mb window::

    mask_score    = sum(wt - mask)
    shuffle_score = sum(wt - shuffle)

i.e. the "absolute difference between the two outputs" of Supplementary Methods
§6.2.2, summed over bins instead of averaged, so the magnitude scales with the
number of bins (1,024 for the CAGE head). The score is signed: a positive value
means the perturbation *reduced* predicted CAGE output.

Label definition
----------------
Supplementary Methods §5.5 defines active CREs by ``tRRA > 2``; this script
binarises at ``RRA > 1`` (see :func:`load_and_merge`) and
:func:`threshold_sensitivity` sweeps 0.5-3.0 around that cut. The manuscript
threshold and this default therefore differ — the sweep exists for that reason.

Expected on disk
----------------
``data/AlphaGenome_FACS_perturbation.h5``
    Written by ``run_facs_perturbation.py``; layout::

        /cage/<sample_id>/wt      float16 (n_bins,)
        /cage/<sample_id>/mask    float16 (n_bins,)
        /cage/<sample_id>/shuffle float16 (n_bins,)
        attrs: chr, start, end, cell_line

``data/FACS_fluorescence_dataset.txt``
    Tab-separated CRE table with at least ``id``, ``cell_line``, ``chr``,
    ``start``, ``end``, ``RRA``. The code calls the activity column ``RRA``
    although the manuscript describes the quantity as tRRA.
``data/FACS_split_{hPSC,K562}_ids.json``
    The CRE-level 7:2:1 test split, shape ``{"test": [{"cre_ids": ...}, ...]}``.

Output artifacts: ``data/AlphaGenome_FACS_perturbation_scores.csv`` plus four
PNGs under ``data/`` (threshold sensitivity, score-vs-RRA scatter, RRA-quintile
plot, example WT/MASK/SHUFFLE tracks).

Known defect
------------
:func:`evaluate_on_test` and :func:`threshold_sensitivity` unpack the return of
``alphagenome_common.m2a_combo_score`` into ``(score, results)``, but that helper
returns a single float. Running this script end-to-end therefore raises
``TypeError: cannot unpack non-iterable float object`` as soon as step 3 starts;
steps 1-2 and their HDF5/CSV products are unaffected.

External dependencies not vendored in this repository: ``h5py``, ``matplotlib``
and ``scipy`` (``alphagenome_common`` is local; this analysis path does not need
the AlphaGenome client or the API keys).
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
# Declared but never read in this script: the FACS runs resume from the HDF5
# itself (see run_facs_perturbation.py), so this path is vestigial.
CHECKPOINT_FILE = 'data/checkpoint_FACS_perturb.pkl'


def compute_scores_from_h5(cre_df=None):
    """Compute mask_score / shuffle_score for every sample in HDF5.

    Parameters
    ----------
    cre_df : pandas.DataFrame or None, optional
        Unused. Present only so that callers can pass the CRE table through
        without branching; all metadata is read from the HDF5 attributes.

    Returns
    -------
    pandas.DataFrame
        One row per HDF5 sample group with columns ``sample_id``
        (``"{id}@{cell_line}"``), ``cell_line``, ``mask_score`` and
        ``shuffle_score``.

    Notes
    -----
    * Each data set is read into ``float32`` before subtraction. The arrays are
      stored as ``float16``; differencing in half precision would quantise the
      small WT-vs-perturbed differences that *are* the signal, so the upcast is
      deliberate.
    * Scores are summed over every bin of the 1 Mb window (the CAGE head returns
      1,024 bins at 1 kb resolution). No window restriction is applied, so
      effects anywhere in the window contribute, not only at the CRE.
    * Reads the whole file group by group instead of loading it at once; the
      FACS dataset's combined arrays would exceed RAM as a single dense tensor.
    """
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
    """Merge scores with the original FACS data, create RRA > 1 binary labels.

    Parameters
    ----------
    score_df : pandas.DataFrame
        Output of :func:`compute_scores_from_h5`.
    cre_csv : str, optional
        Tab-separated FACS CRE table.

    Returns
    -------
    pandas.DataFrame
        Left-joined table (scores kept even if the CRE is missing from the
        reference table, but such rows are then dropped on the null ``RRA``). It
        gains an ``id``/``cl`` split of ``sample_id`` and an integer ``label``.

    Notes
    -----
    * ``sample_id`` is ``"{cre_id}@{cell_line}"``; splitting on ``'@'`` is safe
      only because CRE ids do not contain that character.
    * Rows with a null ``RRA`` are dropped: a CRE with no activity measurement
      cannot be labelled, and silently treating it as negative would inflate
      specificity.
    * The threshold here is ``RRA > 1``, *not* the ``tRRA > 2`` used in
      Supplementary Methods §5.5. Positives are therefore far more numerous than
      the manuscript's 3,592 / 2,253 counts, which is why AUPRC should be read
      alongside AUROC.
    * The merged table is also written to :data:`SCORE_CSV` because the plots
      and any downstream re-analysis reuse it.
    """
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
    """Load FACS test split CRE ids (same as 20260611/run_FACS_benchmark.py).

    Returns
    -------
    list of str
        Test-split identifiers in ``"{cre_id}@{cell_line}"`` form — hPSC ids
        first, then K562 ids, so the list order mirrors the JSON payloads rather
        than any meaningful ordering.

    Notes
    -----
    The split is CRE-identifier level (7:2:1) and is shared with the eScreen and
    other baseline scripts via these JSON files, which is what makes the
    cross-model comparison in Supplementary Methods §6.1 fair: every model is
    scored on exactly the same held-out CREs. The comment references
    ``20260611/run_FACS_benchmark.py``, a path outside this directory that is
    not part of the repository snapshot.
    """
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
    """Evaluate MASK / SHUFFLE strategies on the FACS test split.

    Parameters
    ----------
    merged : pandas.DataFrame
        Output of :func:`load_and_merge`.

    Returns
    -------
    dict
        ``combo_mask``, ``combo_shuffle`` and the two per-cell-line metric lists
        under ``results_mask`` / ``results_shuffle``.

    Notes
    -----
    * ``groups`` is the cell-line column, so AUROC/AUPRC are computed per cell
      line and then reduced with the m2a composite of Supplementary Methods
      §5.5 — a model is never scored by pooling K562 and hPSC, which would mix
      two very different label prevalences.
    * **Currently broken**: it unpacks the float returned by
      ``m2a_combo_score`` into two values and will raise ``TypeError``. See the
      module docstring; the fix would be to call ``m2a_combo_score`` and
      ``per_cell_line_metrics`` separately.
    """
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
    """AUROC / AUPRC changes across different RRA thresholds.

    Parameters
    ----------
    merged : pandas.DataFrame
        Output of :func:`load_and_merge`, restricted internally to the test
        split.

    Returns
    -------
    pandas.DataFrame
        One row per threshold with ``threshold``, ``n_pos``, ``n_neg``,
        ``mask_combo``, ``shuffle_combo``, ``mask_AUROC`` and ``shuffle_AUROC``;
        also written as ``data/FACS_threshold_sensitivity.png``.

    Notes
    -----
    * The threshold sweep exists because the script's default binarisation
      (``RRA > 1``) does not match Supplementary Methods §5.5 (``tRRA > 2``).
      Showing the curve is the honest way to report that choice.
    * A threshold is skipped when it leaves fewer than 10 samples of either
      class: AUROC becomes extremely noisy and AUPRC is dominated by the class
      prior, so the point would be misleading rather than informative.
    * Same ``m2a_combo_score`` unpacking defect as :func:`evaluate_on_test`;
      this function raises before the figure is written.
    """
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
    """Spearman correlation between perturbation score and RRA + scatter plots.

    Parameters
    ----------
    merged : pandas.DataFrame
        Output of :func:`load_and_merge`; **all** FACS samples are used here, not
        only the test split.

    Returns
    -------
    None
        Results are printed and written to
        ``data/FACS_perturbation_vs_RRA.png``.

    Notes
    -----
    * Spearman rather than Pearson: the relationship between an in silico
      perturbation magnitude and tRRA is monotone at best, and both variables
      have heavy tails and outliers that Pearson would be dominated by.
    * Including the non-test samples makes this a descriptive sanity check, not
      a benchmark number — the AUROC/AUPRC in :func:`evaluate_on_test` are the
      reported values.
    * The p-values are computed per cell line without multiple-testing
      correction; with tens of thousands of CREs essentially anything is
      "significant", so read the rho, not the p.
    """
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
    """Mean MASK perturbation score by RRA quintile.

    Parameters
    ----------
    merged : pandas.DataFrame
        Output of :func:`load_and_merge`.

    Returns
    -------
    None
        The figure is written to
        ``data/FACS_perturbation_by_RRA_quintile.png``.

    Notes
    -----
    * ``rank(method='first')`` before ``qcut`` guarantees five equally sized bins
      even though RRA has many ties (many CREs share the same discretised
      value); without the rank, ``qcut`` would raise on duplicate bin edges.
    * Error bars are the standard error of the mean (``sem``), not the standard
      deviation, so they shrink with the (large) per-quintile sample count.
    * Only the MASK score is plotted; the SHUFFLE panel is intentionally left to
      the scatter plot.
    """
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
    """Example: WT / MASK / SHUFFLE CAGE tracks for a sample.

    Parameters
    ----------
    merged : pandas.DataFrame
        Output of :func:`load_and_merge`, used to look up the CRE interval and
        its score for the title.
    sample_id : str
        HDF5 group name, ``"{cre_id}@{cell_line}"``.
    window_bp : int, optional
        Width of the plotted genomic window in bp, centred on the perturbed
        region (default 20 kb).
    save_prefix : str, optional
        Filename prefix; ``"@"`` in the sample id is replaced by ``"_"``.

    Returns
    -------
    None
        The figure is written to ``"{save_prefix}_{sample_id}.png"``.

    Notes
    -----
    * Two different coordinate systems meet here. The stored arrays are indexed
      by *window offset* (0 .. 1,048,575) while the x axis is genomic: the
      conversion is ``x = begin + offset``, where ``begin`` is the HDF5
      ``start`` attribute — i.e. the genomic start of the 1 Mb window, which is
      negative when the window runs off the chromosome start.
    * The shaded CRE span is drawn from ``merged``'s genomic ``start``/``end``,
      so it lines up with the x axis directly and *not* with the array index.
    * The perturbed window is always centred at :data:`PERTURB_CENTER`
      (524,288) in window-local coordinates, hence ``s = center - half``.
    """
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
    """Run the whole FACS analysis chain and print the artifact list.

    Returns
    -------
    None

    Notes
    -----
    Order matters: scores must exist before the merge, and the merge before any
    evaluation or plot. As noted in the module docstring, the run aborts inside
    :func:`evaluate_on_test` (float-unpacking bug), so the plots below it are not
    reached until that is fixed.
    """
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
