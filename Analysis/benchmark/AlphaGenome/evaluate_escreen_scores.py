# -*- coding: utf-8 -*-
"""
eScreen test set — compute perturbation scores from HDF5 and evaluate.

Pipeline (from AlphaGenome_20260527.0.ipynb):
  1. Read wt/mask/shuffle CAGE signals from HDF5
  2. mask_score = sum(WT - MASK), shuffle_score = sum(WT - SHUFFLE)
  3. Merge with cre_list to get labels (per cell line)
  4. Evaluate AUROC / AUPRC per cell line (MASK and SHUFFLE strategies)

Usage:
    python evaluate_escreen_scores.py
"""
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc

from alphagenome_common import per_cell_line_metrics, print_metrics


def load_and_compute_scores_batched(h5_filepath='data/AlphaGenome_CAGE_output_eScreenTest_1.h5',
                                    batch_size=1000,
                                    output_csv='data/cage_perturbation_scores.csv'):
    """
    Read the HDF5 file in batches and compute scores, avoiding OOM.
    mask_score = sum(WT - MASK), shuffle_score = sum(WT - SHUFFLE)
    """
    with h5py.File(h5_filepath, 'r') as f:
        sample_ids = list(f['cage'].keys())

    total_samples = len(sample_ids)
    print(f"Total samples: {total_samples}")

    n_batches = (total_samples + batch_size - 1) // batch_size
    results = []
    header_written = False

    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)
        batch_ids = sample_ids[start_idx:end_idx]

        batch_results = []
        with h5py.File(h5_filepath, 'r') as f:
            cage_group = f['cage']
            for sample_id in batch_ids:
                try:
                    sample_group = cage_group[sample_id]
                    wt_cage = sample_group['wt'][:]
                    mask_cage = sample_group['mask'][:]
                    shuffle_cage = sample_group['shuffle'][:]

                    mask_score = np.sum(wt_cage.astype(np.float32) - mask_cage.astype(np.float32))
                    shuffle_score = np.sum(wt_cage.astype(np.float32) - shuffle_cage.astype(np.float32))

                    batch_results.append({
                        'sample_id': sample_id,
                        'mask_score': mask_score,
                        'shuffle_score': shuffle_score
                    })
                except Exception as e:
                    print(f"Error processing {sample_id}: {e}")
                    continue

        results.extend(batch_results)

        # Periodically write to CSV (avoid memory accumulation)
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            batch_df = pd.DataFrame(batch_results)
            if not header_written:
                batch_df.to_csv(output_csv, index=False, mode='w')
                header_written = True
            else:
                batch_df.to_csv(output_csv, index=False, mode='a', header=False)
            batch_results = []
            gc.collect()

    return pd.DataFrame(results)


def load_and_compute_scores_streaming(h5_filepath='data/AlphaGenome_CAGE_output_eScreenTest_1.h5',
                                      output_csv='data/cage_perturbation_scores.csv'):
    """Streaming version — minimal memory, write each sample to file immediately."""
    import csv

    with h5py.File(h5_filepath, 'r') as f:
        cage_group = f['cage']
        sample_ids = list(cage_group.keys())
        total_samples = len(sample_ids)
        print(f"Total samples: {total_samples}")

        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['sample_id', 'mask_score', 'shuffle_score'])

            written = 0
            for sample_id in tqdm(sample_ids, desc="Processing samples"):
                try:
                    sample_group = cage_group[sample_id]
                    wt_cage = sample_group['wt'][:]
                    mask_cage = sample_group['mask'][:]
                    shuffle_cage = sample_group['shuffle'][:]

                    mask_score = np.sum(wt_cage.astype(np.float32) - mask_cage.astype(np.float32))
                    shuffle_score = np.sum(wt_cage.astype(np.float32) - shuffle_cage.astype(np.float32))

                    writer.writerow([sample_id, mask_score, shuffle_score])
                    written += 1
                except Exception as e:
                    print(f"Error processing {sample_id}: {e}")
                    continue

                # Flush every 1000 samples
                if written % 1000 == 0:
                    csvfile.flush()

    print(f"Results saved to {output_csv}")
    return output_csv


def analyze_in_chunks(h5_filepath='data/AlphaGenome_CAGE_output_eScreenTest_1.h5',
                      chunk_size=10000):
    """Chunked analysis: compute statistics only, without keeping all data."""
    with h5py.File(h5_filepath, 'r') as f:
        cage_group = f['cage']
        sample_ids = list(cage_group.keys())
        total_samples = len(sample_ids)

        mask_sum = 0
        mask_sumsq = 0
        shuffle_sum = 0
        shuffle_sumsq = 0
        min_mask = float('inf')
        max_mask = float('-inf')
        min_shuffle = float('inf')
        max_shuffle = float('-inf')

        for i in tqdm(range(0, total_samples, chunk_size), desc="Analyzing chunks"):
            chunk_ids = sample_ids[i:i + chunk_size]
            chunk_mask_scores = []
            chunk_shuffle_scores = []

            for sample_id in chunk_ids:
                try:
                    sample_group = cage_group[sample_id]
                    wt_cage = sample_group['wt'][:]
                    mask_cage = sample_group['mask'][:]
                    shuffle_cage = sample_group['shuffle'][:]

                    mask_score = np.sum(wt_cage.astype(np.float32) - mask_cage.astype(np.float32))
                    shuffle_score = np.sum(wt_cage.astype(np.float32) - shuffle_cage.astype(np.float32))

                    chunk_mask_scores.append(mask_score)
                    chunk_shuffle_scores.append(shuffle_score)
                except Exception as e:
                    print(f"Error processing {sample_id}: {e}")
                    continue

            if chunk_mask_scores:
                chunk_mask = np.array(chunk_mask_scores)
                chunk_shuffle = np.array(chunk_shuffle_scores)

                mask_sum += np.sum(chunk_mask)
                mask_sumsq += np.sum(chunk_mask ** 2)
                shuffle_sum += np.sum(chunk_shuffle)
                shuffle_sumsq += np.sum(chunk_shuffle ** 2)

                min_mask = min(min_mask, np.min(chunk_mask))
                max_mask = max(max_mask, np.max(chunk_mask))
                min_shuffle = min(min_shuffle, np.min(chunk_shuffle))
                max_shuffle = max(max_shuffle, np.max(chunk_shuffle))

            del chunk_mask_scores, chunk_shuffle_scores
            gc.collect()

        n = len(sample_ids)
        mask_mean = mask_sum / n
        mask_std = np.sqrt(mask_sumsq / n - mask_mean ** 2)
        shuffle_mean = shuffle_sum / n
        shuffle_std = np.sqrt(shuffle_sumsq / n - shuffle_mean ** 2)

        return {
            'n_samples': n,
            'mask_score': {'mean': mask_mean, 'std': mask_std, 'min': min_mask, 'max': max_mask},
            'shuffle_score': {'mean': shuffle_mean, 'std': shuffle_std,
                              'min': min_shuffle, 'max': max_shuffle}
        }


def sample_generator(h5_filepath='data/AlphaGenome_CAGE_output_eScreenTest_1.h5'):
    """Generator yielding samples one by one for custom processing."""
    with h5py.File(h5_filepath, 'r') as f:
        cage_group = f['cage']
        for sample_id in cage_group.keys():
            sample_group = cage_group[sample_id]
            yield {
                'sample_id': sample_id,
                'wt_cage': sample_group['wt'][:],
                'mask_cage': sample_group['mask'][:],
                'shuffle_cage': sample_group['shuffle'][:]
            }


def evaluate(score_df, cre_list):
    """Merge scores with CRE labels and evaluate MASK / SHUFFLE strategies."""
    # Parse sample_id -> id + cell_line
    score_df[['id', 'cell_line']] = score_df['sample_id'].str.split('@', expand=True).values
    score_df['cell_line'] = score_df['cell_line'].replace('WTC11', 'hPSC')

    df = pd.merge(score_df, cre_list, on=['id', 'cell_line'], how='right')
    df = df.dropna(subset=['mask_score', 'shuffle_score', 'label'])
    print(f'Merged samples: {len(df)}')

    # MASK strategy
    print('\n=== MASK strategy ===')
    results = per_cell_line_metrics(df['mask_score'], df['label'], df['cell_line'])
    print_metrics(results, 'MASK strategy')

    # SHUFFLE strategy
    print('\n=== SHUFFLE strategy ===')
    results = per_cell_line_metrics(df['shuffle_score'], df['label'], df['cell_line'])
    mean_auroc, mean_auprc = print_metrics(results, 'SHUFFLE strategy')
    print(f'\nMean AUROC: {mean_auroc:.4f}, Mean AUPRC: {mean_auprc:.4f}')

    return df


if __name__ == '__main__':
    h5_filepath = 'data/AlphaGenome_CAGE_output_eScreenTest_1.h5'

    # Option 1: batched processing in memory (default)
    df = load_and_compute_scores_batched(h5_filepath, batch_size=5000)
    print(f"Loaded {len(df)} samples")
    print(df.head())

    # Option 2: streaming to file only (for tight memory)
    # output_file = load_and_compute_scores_streaming(h5_filepath,
    #                                                 'data/cage_perturbation_scores.csv')

    # Option 3: statistics only
    # stats = analyze_in_chunks(h5_filepath)

    # Option 4: generator-based processing
    # for i, sample_data in enumerate(sample_generator(h5_filepath)):
    #     if i >= 10:
    #         break

    cre_list = pd.read_csv('data/p0_filtered_samples.csv')
    evaluate(df, cre_list)
