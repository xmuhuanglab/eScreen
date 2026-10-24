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

What this script implements
---------------------------
The AlphaGenome in silico perturbation arm of Supplementary Methods §6.2.2, run
on the eScreen benchmark test set: for every held-out CRE, a 1 Mb window centred
on the CRE summit is submitted three times to the hosted AlphaGenome API — once
unperturbed (WT), once with the central 512 bp replaced by ``N`` (MASK), and once
with the central 512 bp shuffled (SHUFFLE) — requesting the CAGE output head for
the sample's cell-line ontology term.

Deviations from the manuscript, stated plainly
----------------------------------------------
* §6.2.2 describes the perturbation readout as the **RNA-seq** tracks; this code
  requests **CAGE** (``OutputType.CAGE`` in ``alphagenome_common``). §5.5
  describes the same perturbation experiment with a CAGE readout, so the code
  follows §5.5. The manuscript is internally inconsistent here.
* §6.2.2 says the *absolute* difference between WT and perturbed output is used;
  the score computed downstream is the **signed** sum ``sum(WT - ALT)``.
* §5.5 specifies a 1 Mb window "centred on the CRE summit" and a central 512 bp
  perturbation; both are implemented exactly.

AlphaGenome is not vendored here: inference goes through the ``alphagenome``
package against the hosted API (hard-coded keys in ``alphagenome_common``). No
local checkpoint is used.

Expected on disk
----------------
``data/p0_filtered_samples.csv`` (input)
    eScreen benchmark table; only rows with ``split == 'test'`` are used, and
    ``start``/``end`` must be the CRE interval on hg38 (0-based, half-open) so
    that the summit can be computed as ``(start + end) // 2``.
``data/hg38.fa`` (input)
    Indexed hg38 for :mod:`pyfaidx`.
``data/AlphaGenome_CAGE_output_eScreenTest_1.h5`` (output)
    ``/cage/<id>@<cell_line>/{wt,mask,shuffle}`` (``float16`` per CAGE bin) with
    attributes ``chromosome``, ``start``, ``end``, ``cell_line``. Written in
    append mode, so the script is resumable by re-running it.

Order of use: run this script first, then ``evaluate_escreen_scores.py``.

Notes on the run
----------------
* The per-sample ``float16`` CAGE vector is ``(n_bins,)`` — the CAGE tracks are
  collapsed with ``.sum(axis=1)`` before storage, so all per-track information is
  discarded and only the per-bin total survives.
* The HDF5 attribute here is called ``chromosome`` (this file) whereas
  ``run_facs_perturbation.py`` writes ``chr``. Consumers must match the producer;
  ``analyze_facs_scores.py`` reads ``chr`` and would fail on a file written here.
* ``CheckpointManager`` is imported but never instantiated: resumption is driven
  entirely by the groups already present in the HDF5. The import is dead.
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
    """Load the CRE list (test split only), build a 1 Mb window per CRE.

    Parameters
    ----------
    csv_path : str, optional
        eScreen benchmark table exported by ``export_p0_samples.py``. Must have
        ``split``, ``start``, ``end``, ``cell_line`` and ``id``.

    Returns
    -------
    pandas.DataFrame
        The test rows with two added columns, ``begin`` and ``stop``: the
        0-based half-open bounds of a 1,048,576 bp window centred on the CRE
        summit. ``begin`` can be negative and ``stop`` can exceed the chromosome
        length — ``alphagenome_common.prepare_sequences_fast`` clips and pads.

    Notes
    -----
    * The summit is approximated by the interval midpoint
      ``(start + end) // 2``. Supplementary Methods §5.5 says "centred on the CRE
      summit"; the true summit (the DNase/ATAC max) is not stored in this table,
      so the midpoint is used instead. For a 512 bp window this shifts the
      perturbed bases by at most a few hundred bp relative to a true summit.
    * ``HALF = 1048576 // 2`` is a *flank* of 524,288 bp, so the resulting window
      is exactly 1,048,576 bp long and the perturbation offsets in
      ``alphagenome_common`` (centred on 524,288) land on the CRE midpoint.
    * Only the test split is used, matching the held-out evaluation in
      Supplementary Methods §6.1.
    """
    cre_list = pd.read_csv(csv_path)
    cre_list = cre_list[cre_list['split'] == 'test']
    HALF = 1048576 // 2
    summit = (cre_list['start'] + cre_list['end']) // 2
    cre_list['begin'] = summit - HALF
    cre_list['stop']  = summit + HALF
    print(f'Test CREs: {len(cre_list)}, cell lines: {dict(cre_list["cell_line"].value_counts())}')
    return cre_list


def save_results_flat(h5file, results, global_idx, skip_existing=True):
    """Save results to HDF5 (one group per sample: wt/mask/shuffle + metadata).

    Parameters
    ----------
    h5file : h5py.File
        Open file in append mode; a ``cage`` group must already exist.
    results : list of tuple
        Output of ``alphagenome_common.process_batch_parallel``:
        ``(sample_id, wt_cage, mask_cage, shuffle_cage, row)``.
    global_idx : list of one int
        Mutable counter (a one-element list, used as a poor man's ``nonlocal``)
        incremented per sample actually written.
    skip_existing : bool, optional
        If True, groups that already exist are left untouched — this is the
        resume guard, and it makes the function idempotent for a partially
        written batch.

    Returns
    -------
    int
        Number of samples newly written (``0`` for an entirely duplicated
        batch).

    Notes
    -----
    * Datasets are stored without compression: the arrays are small and the run
      is API-latency bound, so CPU spent on compression would only slow it down.
    * ``attrs['chromosome']`` is this file's spelling; the FACS runner writes
      ``attrs['chr']`` instead. Note the asymmetry when reading either file.
    * ``rw.id``/``rw.cell_line`` are read as Series attributes (pandas resolves
      them against the column names), which requires those columns to exist and
      not clash with Series methods.
    """
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
    """Main loop: predict WT/MASK/SHUFFLE per batch into HDF5, resumable.

    Parameters
    ----------
    cre_list : pandas.DataFrame
        Test-split rows with ``begin``/``stop`` (see :func:`build_cre_list`).
    hg38 : pyfaidx.Fasta
        Indexed hg38.
    chrom_len : dict of str to int
        Chromosome lengths.
    model1, model2, model3 : DnaClient
        Clients bound to the three API keys (WT / MASK / SHUFFLE).
    filepath : str, optional
        Output HDF5; opened in append mode when it already exists.

    Returns
    -------
    None

    Notes
    -----
    * **Resumption** is derived from the HDF5 itself: existing group keys give
      the set of finished samples, and only the remaining row indices are
      batched. The checkpoint machinery living in ``alphagenome_common`` is not
      used here (the import is dead code), so deleting the HDF5 restarts the run.
    * **Batching** is done BY CELL LINE. The loop breaks a batch whenever
      ``cell_line`` changes, which is required because
      ``process_batch_parallel`` takes the ontology term from the first row of
      the batch and assumes the whole batch shares it. A consequence is that the
      final batch of each cell line can be smaller than ``BATCH_SIZE``, and
      ``row['cell_line'] != current_cell_line`` is only checked *before* appending
      the current row, so the very first row of a new cell line starts a new batch
      correctly.
    * Batches are taken in ``pending_cre.loc[batch_indices]`` order, and the
      indices come from a ``set`` difference, so the processing order (though not
      the result) varies between runs.
    * Errors are caught per batch and only printed; the loop continues, and
      nothing records the failure. A failed batch is simply retried on the next
      invocation because its groups are still missing from the HDF5.
    * ``processed_count`` counts samples *attempted*, while ``total_new_samples``
      counts groups actually created (which can be fewer if a group already
      existed). The two are printed together and can legitimately differ.
    """
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
