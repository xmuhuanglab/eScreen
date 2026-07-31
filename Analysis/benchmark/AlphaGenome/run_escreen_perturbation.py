# -*- coding: utf-8 -*-
"""
AlphaGenome CAGE perturbation prediction — eScreen test set (5 cell lines).

Pipeline (from AlphaGenome_20260527.0.ipynb):
  1. Load the test split of p0_filtered_samples.csv, take a 1 Mb window
     centered at the CRE summit
  2. Build WT / MASK / SHUFFLE sequences (fixed 512 bp window at the center)
  3. Predict CAGE with three models in parallel, write batches to HDF5
     (resumable)

Usage:
    python run_escreen_perturbation.py
"""
import os
import time
import gc
import h5py
import pandas as pd
from tqdm import tqdm

from alphagenome_common import (
    create_models, load_genome, ONTOLOGY_TERM,
    BATCH_SIZE, prepare_sequences_fast, predict_single_model,
    process_batch_parallel, CheckpointManager,
)


def build_cre_list(csv_path='data/p0_filtered_samples.csv'):
    """Load the CRE list (test split only), build a 1 Mb window per CRE."""
    cre_list = pd.read_csv(csv_path)
    cre_list = cre_list[cre_list['split'] == 'test']
    HALF = 1048576 // 2
    summit = (cre_list['start'] + cre_list['end']) // 2
    cre_list['begin'] = summit - HALF
    cre_list['stop']  = summit + HALF
    print(f'Test CREs: {len(cre_list)}, cell lines: {dict(cre_list["cell_line"].value_counts())}')
    return cre_list


def save_results_flat(h5file, results, global_idx, skip_existing=True):
    """Save results to HDF5 (one group per sample: wt/mask/shuffle + metadata)."""
    cage_group = h5file['cage']

    saved_count = 0
    for sample_id, wt_cage, mask_cage, shuffle_cage, rw in results:
        if skip_existing and sample_id in cage_group:
            print(f"  Skipping existing sample: {sample_id}")
            continue

        sample_group = cage_group.create_group(sample_id)
        sample_group.create_dataset('wt', data=wt_cage, compression=None)
        sample_group.create_dataset('mask', data=mask_cage, compression=None)
        sample_group.create_dataset('shuffle', data=shuffle_cage, compression=None)

        sample_group.attrs['chromosome'] = str(rw.chr)
        sample_group.attrs['start'] = rw.begin
        sample_group.attrs['end'] = rw.stop
        sample_group.attrs['cell_line'] = rw.cell_line

        saved_count += 1
        global_idx[0] += 1

    return saved_count


def main(cre_list, hg38, chrom_len, model1, model2, model3,
         filepath='data/AlphaGenome_CAGE_output_eScreenTest_1.h5'):
    """Main loop: predict WT/MASK/SHUFFLE per batch into HDF5, resumable."""
    # Open HDF5 (append mode)
    if os.path.exists(filepath):
        f = h5py.File(filepath, 'a')
        if 'cage' not in f:
            f.create_group('cage')
        print(f"Resuming from existing file: {filepath}")
    else:
        f = h5py.File(filepath, 'w')
        f.create_group('cage')
        print(f"Creating new file: {filepath}")

    # Already processed samples
    existing_samples = set(f['cage'].keys())
    print(f"Found {len(existing_samples)} already processed samples")

    # Filter out processed samples
    all_indices = set(range(len(cre_list)))
    processed_indices = set()
    for idx, row in cre_list.iterrows():
        sample_id = f"{row.id}@{row.cell_line}"
        if sample_id in existing_samples:
            processed_indices.add(idx)

    pending_indices = list(all_indices - processed_indices)
    if not pending_indices:
        print("All samples already processed!")
        f.close()
        return

    print(f"Processing {len(pending_indices)} new samples...")

    # Group by cell line and build batches
    pending_cre = cre_list.iloc[pending_indices].copy().sort_values('cell_line')

    batches = []
    current_batch = []
    current_cell_line = None
    for idx, row in pending_cre.iterrows():
        if current_cell_line is not None and row['cell_line'] != current_cell_line:
            if current_batch:
                batches.append(current_batch.copy())
                current_batch = []
        current_batch.append(idx)
        current_cell_line = row['cell_line']
        if len(current_batch) >= BATCH_SIZE:
            batches.append(current_batch.copy())
            current_batch = []
    if current_batch:
        batches.append(current_batch)

    print(f"Total batches: {len(batches)}, Avg size: {len(pending_indices)/len(batches):.1f}")

    # Run prediction
    t0 = time.time()
    processed_count = 0
    total_new_samples = 0

    try:
        with tqdm(total=len(pending_indices), desc="Processing", unit="sample") as pbar:
            for batch_idx, batch_indices in enumerate(batches, 1):
                batch_rows = pending_cre.loc[batch_indices]

                try:
                    # Parallel WT / MASK / SHUFFLE prediction
                    results = process_batch_parallel(
                        model1, model2, model3, batch_rows, hg38, chrom_len
                    )

                    # Save results (skip samples that already exist)
                    if results:
                        saved_count = save_results_flat(f, results, [total_new_samples],
                                                        skip_existing=True)
                        total_new_samples += saved_count
                        processed_count += len(results)
                        pbar.update(len(results))
                    else:
                        pbar.update(len(batch_rows))

                    elapsed = time.time() - t0
                    speed = processed_count / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'speed': f'{speed:.2f}samp/s',
                        'batch': f'{batch_idx}/{len(batches)}',
                        'new': len(results) if results else 0,
                    })

                    # Periodic flush and GC
                    if batch_idx % 10 == 0:
                        f.flush()
                        gc.collect()

                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        elapsed_total = time.time() - t0
        print("\n" + "=" * 50)
        print("COMPLETED SUCCESSFULLY!")
        print(f"Total time: {elapsed_total:.2f} seconds")
        print(f"New samples processed: {total_new_samples}/{len(pending_indices)}")
        print(f"Average speed: {processed_count/elapsed_total:.2f} samples/second")
        print("=" * 50)

    except KeyboardInterrupt:
        print("\n" + "=" * 50)
        print("INTERRUPTED by user!")
        print(f"Progress saved: {total_new_samples}/{len(pending_indices)}")
        print("=" * 50)

    finally:
        f.flush()
        f.close()
        print("\nFile saved successfully")


if __name__ == '__main__':
    model1, model2, model3 = create_models()
    hg38, chrom_len = load_genome('data/hg38.fa')
    cre_list = build_cre_list('data/p0_filtered_samples.csv')
    main(cre_list, hg38, chrom_len, model1, model2, model3,
         filepath='data/AlphaGenome_CAGE_output_eScreenTest_1.h5')
