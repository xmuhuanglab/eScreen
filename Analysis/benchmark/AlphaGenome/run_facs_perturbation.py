# -*- coding: utf-8 -*-
"""
AlphaGenome CAGE perturbation prediction — FACS dataset (K562 + hPSC).

Pipeline (from AlphaGenome_perturbation_FACS.ipynb):
  1. Load FACS_fluorescence_dataset.txt, take a 1 Mb window centered at
     the CRE summit
  2. Build WT / MASK / SHUFFLE sequences (fixed 512 bp window at the center)
  3. Predict CAGE with three models in parallel, write batches to HDF5
     (resumable, Ctrl+C to interrupt and resume later)

Usage:
    python run_facs_perturbation.py

What this script implements
---------------------------
Generation of the AlphaGenome perturbation predictions used by the FACS
readout-specific evaluation in Supplementary Methods §5.5 (K562 and hPSC). For
each CRE, a 1 Mb window centred on the CRE summit is sent to the hosted
AlphaGenome API three times — unperturbed (WT), with the central 512 bp masked by
``N`` (MASK), and with the central 512 bp shuffled (SHUFFLE) — using the CAGE
output head and the cell line's ontology term.

This matches §5.5's description ("a 1 Mb genomic window centered on the CRE
summit was extracted, and its central 512 bp ... was either replaced with Ns
(masking) or randomly shuffled"). Note that §6.2.2 describes the same experiment
with an RNA-seq readout, while both the manuscript's §5.5 text and this code use
the CAGE head.

AlphaGenome is not vendored: inference is remote, through the ``alphagenome``
package and the hard-coded API keys in ``alphagenome_common``. No local
checkpoint exists.

Expected on disk
----------------
``data/FACS_fluorescence_dataset.txt`` (input)
    Tab-separated FACS CRE table with ``id``, ``chr``, ``start``, ``end``,
    ``cell_line``, ``RRA``. ``start``/``end`` are the CRE interval (0-based,
    half-open) and the summit is approximated by the midpoint.
``data/hg38.fa`` (input)
    Indexed hg38 for :mod:`pyfaidx`.
``data/AlphaGenome_FACS_perturbation.h5`` (output)
    ``/cage/<id>@<cell_line>/{wt,mask,shuffle}`` (``float16``, one value per CAGE
    bin) with attributes ``chr``, ``start``, ``end``, ``cell_line``.
``data/checkpoint_FACS_perturb.pkl`` (output, transient)
    Pickled ``processed``/``failed`` index sets; deleted automatically once every
    sample is done.

Order of use: run this script first, then ``analyze_facs_scores.py``.

Deviations and rough edges
--------------------------
* The HDF5 attribute is ``chr`` here but ``chromosome`` in
  ``run_escreen_perturbation.py``; the two producers are not interchangeable.
* ``CheckpointManager.mark_processed`` is called with ``rw.name`` — the pandas
  *index label*, not the CRE id. It happens to be the correct integer because the
  table is read with the default ``RangeIndex`` and re-indexed with
  ``cre_df.iloc[pending_idx]``; it would silently record the wrong indices if the
  input file ever carried a non-default index.
* If there is nothing left to do, the early branch prints "All samples already
  processed!" and returns without closing the HDF5 handle (the ``finally`` only
  guards the prediction branch), so on Windows the subsequent checkpoint cleanup
  can fail with a sharing violation. Closing the file before ``cleanup`` would
  fix it.
"""
import os
import time
import gc
import h5py
import pandas as pd
from tqdm import tqdm

from alphagenome_common import (
    create_models, load_genome, check_ontology,
    BATCH_SIZE, PERTURB_START, PERTURB_END, WINDOW_SIZE,
    process_batch_parallel, CheckpointManager,
)


OUTPUT_H5 = 'data/AlphaGenome_FACS_perturbation.h5'
CHECKPOINT_FILE = 'data/checkpoint_FACS_perturb.pkl'


def build_cre_df(csv_path='data/FACS_fluorescence_dataset.txt', sep='\t'):
    """Load the FACS dataset, build a 1 Mb window per CRE.

    Parameters
    ----------
    csv_path : str, optional
        Tab-separated FACS CRE table.
    sep : str, optional
        Field separator; tab for the shipped file.

    Returns
    -------
    pandas.DataFrame
        The input table plus ``begin`` / ``stop``: 0-based half-open bounds of a
        1,048,576 bp window centred on the CRE midpoint. Bounds are unclipped and
        may be negative or past the chromosome end; padding happens later.

    Notes
    -----
    * The summit is taken as ``(start + end) // 2`` (integer division, so the
      window start is biased one bp left for odd-length CREs). Supplementary
      Methods §5.5 says "centred on the CRE summit"; the midpoint is used because
      no summit column is stored in this table.
    * ``.astype(int)`` is applied to the midpoint because an all-NaN or float
      column would otherwise propagate floats into the row ``begin``/``stop``
      values and from there into HDF5 attributes.
    """
    cre_df = pd.read_csv(csv_path, sep=sep)
    print(f'Total CREs: {len(cre_df)}')
    print(cre_df['cell_line'].value_counts())

    HALF = 1048576 // 2
    summit = ((cre_df['start'] + cre_df['end']) // 2).astype(int)
    cre_df['begin'] = summit - HALF
    cre_df['stop']  = summit + HALF
    return cre_df


def main():
    """Main loop: predict WT/MASK/SHUFFLE per batch into HDF5, resumable.

    Returns
    -------
    None

    Notes
    -----
    * Resumption is doubly guarded: the HDF5 group keys are re-synced into the
      checkpoint at start-up (so a lost pickle costs nothing), and
      ``get_pending_indices`` returns the still-failed indices first, so a
      resumed run retries the samples that broke before spending API quota on
      fresh ones.
    * Batching is grouped by cell line for the same reason as in
      ``run_escreen_perturbation.py``: the ontology term is taken from the first
      row of the batch and assumed homogeneous.
    * ``mark_processed(rw.name)`` records the pandas index label. It coincides
      with the row position only because the table is read with a default
      ``RangeIndex``; see the module docstring.
    * The keyboard-interrupt handler saves progress by relying on the checkpoint
      writes already performed; there is no final flush of unsaved samples
      because each sample is written to HDF5 as soon as its batch returns.
    * ``CheckpointManager.cleanup`` runs after the file is closed, which is what
      lets the pickle be deleted on Windows when the run completes.
    """
    # Create models
    model1, model2, model3 = create_models()
    print('Models created:', model1, model2, model3)

    # Reference genome
    hg38, chrom_len = load_genome('data/hg38.fa')
    print('Chromosomes loaded:', len(chrom_len))

    # FACS dataset
    cre_df = build_cre_df('data/FACS_fluorescence_dataset.txt', sep='\t')

    # Ontology check
    check_ontology(cre_df['cell_line'].unique())

    print(f'Perturbation window: [{PERTURB_START}, {PERTURB_END}) = {WINDOW_SIZE} bp')

    total_samples = len(cre_df)
    ckpt = CheckpointManager(CHECKPOINT_FILE)

    # Open HDF5 (append mode)
    if os.path.exists(OUTPUT_H5):
        f = h5py.File(OUTPUT_H5, 'a')
        if 'cage' not in f:
            f.create_group('cage')
        existing = set(f['cage'].keys())
        print(f'Existing HDF5: {len(existing)} samples already saved')
    else:
        f = h5py.File(OUTPUT_H5, 'w')
        f.create_group('cage')
        existing = set()
        print('New HDF5 created')

    # Sync already-processed samples into the checkpoint
    for idx, row in cre_df.iterrows():
        sid = f"{row.id}@{row.cell_line}"
        if sid in existing:
            ckpt.processed.add(idx)

    # Pending indices
    pending_idx = ckpt.get_pending_indices(total_samples)
    print(f'Pending: {len(pending_idx)} / {total_samples}')

    if not pending_idx:
        print('All samples already processed!')
    else:
        # Sort by cell line and build batches
        pending_df = cre_df.iloc[pending_idx].copy().sort_values('cell_line')
        batches = []
        curr = []
        curr_cl = None
        for idx, row in pending_df.iterrows():
            if curr_cl is not None and row.cell_line != curr_cl and curr:
                batches.append(curr)
                curr = []
            curr.append(idx)
            curr_cl = row.cell_line
            if len(curr) >= BATCH_SIZE:
                batches.append(curr)
                curr = []
        if curr:
            batches.append(curr)

        print(f'Batches: {len(batches)}, avg {len(pending_idx)/len(batches):.1f} samples/batch')

        # Run prediction
        t0 = time.time()
        total_new = 0

        try:
            with tqdm(total=len(pending_idx), desc='Predicting', unit='samp') as pbar:
                for bi, batch_idx_list in enumerate(batches, 1):
                    batch_rows = pending_df.loc[batch_idx_list]

                    try:
                        results = process_batch_parallel(
                            model1, model2, model3, batch_rows, hg38, chrom_len
                        )

                        for sample_id, wt, mask, shuffle, rw in results:
                            grp = f['cage'].create_group(sample_id)
                            grp.create_dataset('wt', data=wt, compression=None)
                            grp.create_dataset('mask', data=mask, compression=None)
                            grp.create_dataset('shuffle', data=shuffle, compression=None)
                            grp.attrs['chr'] = rw.chr
                            grp.attrs['start'] = rw.begin
                            grp.attrs['end'] = rw.stop
                            grp.attrs['cell_line'] = rw.cell_line

                            ckpt.mark_processed(rw.name)
                            total_new += 1
                            pbar.update(1)

                        elapsed = time.time() - t0
                        pbar.set_postfix({'batch': f'{bi}/{len(batches)}', 'new': total_new,
                                          'speed': f'{total_new/elapsed:.2f}/s' if elapsed else '...'})

                        if bi % 10 == 0:
                            f.flush()
                            gc.collect()

                    except Exception as e:
                        print(f'\nBatch {bi} error: {e}')
                        import traceback
                        traceback.print_exc()
                        for idx in batch_idx_list:
                            ckpt.mark_failed(idx)
                        continue

            elapsed = time.time() - t0
            print(f'\nCompleted: {total_new} new samples in {elapsed:.1f}s '
                  f'({elapsed/max(total_new,1):.2f}s/sample)')

        except KeyboardInterrupt:
            print(f'\nInterrupted at {total_new} new samples — checkpoint saved')

        finally:
            f.flush()
            f.close()
            print('HDF5 file closed')

    # Remove checkpoint when everything is done
    ckpt.cleanup(total_samples)
    print(f'\nHDF5 saved to: {OUTPUT_H5}')


if __name__ == '__main__':
    main()
