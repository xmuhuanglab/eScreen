# -*- coding: utf-8 -*-
"""
AlphaGenome common module — code shared by both notebooks:
model creation / reference genome loading / ontology mapping / perturbation
parameters / sequence construction / parallel prediction / checkpoint /
string perturbation utilities / evaluation metrics.

Source:
  - benchmark/AlphaGenome_20260527.0.ipynb
  - benchmark/AlphaGenome_perturbation_FACS.ipynb

What this module implements
---------------------------
The AlphaGenome arm of the model benchmarking described in Supplementary
Methods §6.2.2, together with the AlphaGenome half of the FACS
readout-specific evaluation in Supplementary Methods §5.5.

AlphaGenome is *not* reimplemented here. The model is DeepMind's released
package (``alphagenome``, imported below) and inference runs against their
hosted API, so this repository only wraps it: it builds the three input
sequences per CRE, fires them at the API, and stores the returned tracks. No
AlphaGenome weights are distributed with this repo — Supplementary Methods
§6.2.2 states that embedding-based and fine-tuning experiments were excluded
for licensing reasons, and that only in silico perturbation was run.

Two perturbation strategies are produced for every CRE (both from
Supplementary Methods §6.2.2):

``MASK``
    The 512 bp window at the centre of the 1 Mb input is replaced with ``N``
    ("N-masking perturbation"). Implemented by
    :func:`prepare_sequences_fast` / :func:`mask_in_window`.
``SHUFFLE``
    The same 512 bp window is shuffled in place ("Sequence shuffling
    perturbation"). Implemented by :func:`prepare_sequences_fast` /
    :func:`shuffle_in_window`.

The score actually consumed by the downstream evaluation scripts is
``sum(WT - ALT)`` accumulated over all output bins of the CAGE head; see
``analyze_facs_scores.py`` and ``evaluate_escreen_scores.py``.

Coordinate conventions used throughout
--------------------------------------
Everything is 0-based, half-open (``[start, end)``) on hg38, matching
:mod:`pyfaidx` slicing. The model input is a 1,048,576 bp window centred on the
CRE summit; the perturbed window is the central 512 bp,
``[524,032, 524,544)`` in window-local coordinates (see the constants below).

Expected on disk
----------------
``data/hg38.fa``
    hg38 reference genome indexed for :mod:`pyfaidx` (i.e. ``hg38.fa.fai``
    must sit next to it). Only chr1-22, chrX and chrY are loaded. The path is
    a hard-coded default; adjust before rerunning on another machine.

Nothing else is read from disk. AlphaGenome runs behind the hosted API, so
there is no local checkpoint; the API keys are hard-coded (see the note above
:data:`API_KEY_1`).

External dependencies not vendored in this repository
-----------------------------------------------------
``alphagenome`` (DeepMind), ``pyfaidx``, ``h5py``, ``scikit-learn``,
``scipy`` and ``matplotlib``.
"""
from alphagenome import colab_utils
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client
from alphagenome.models import variant_scorers
from alphagenome.visualization import plot_components
from alphagenome.models.dna_client import OutputType

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm
import h5py
import time
import gc
import os
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.interpolate import PchipInterpolator
import warnings
warnings.filterwarnings('ignore')


# ── AlphaGenome API keys + model creation ────────────────────────
# Hard-coded credentials for the hosted AlphaGenome API. Three separate keys
# are used so that the WT / MASK / SHUFFLE requests of one batch can run
# concurrently without sharing a per-key rate limit (see
# :func:`process_batch_parallel`). These are research credentials committed by
# the authors; treat them as secrets and replace with your own keys (or an
# environment variable) before rerunning.
API_KEY_1 = 'AIzaSyCqc-MYn0DnNcszFh8AjISkgHiiZk8wAUM'
API_KEY_2 = 'AIzaSyC6jslFT4qt9Tbo0L9Ote7m60U__pP68r8'
API_KEY_3 = 'AIzaSyC8xemKYuNnzX0nOnG5dpf6e8B_EMeT_Rc'


def create_models():
    """Create three models for parallel WT / MASK / SHUFFLE prediction.

    Each returned client is bound to its own API key so that the three
    concurrent requests issued per batch do not contend for one key's quota.

    Returns
    -------
    tuple of alphagenome.models.dna_client.DnaClient
        ``(model1, model2, model3)``, backed by :data:`API_KEY_1`,
        :data:`API_KEY_2` and :data:`API_KEY_3` respectively.

    Notes
    -----
    Creating a client performs no inference; models are thin remote handles,
    not local networks. Keep the same three objects alive for the whole run
    (the runner scripts create them once in ``__main__`` and pass them down).
    """
    model1 = dna_client.create(API_KEY_1)
    model2 = dna_client.create(API_KEY_2)
    model3 = dna_client.create(API_KEY_3)
    return model1, model2, model3


# ── Reference genome ─────────────────────────────────────────────
def load_genome(fasta_path='data/hg38.fa'):
    """Open an indexed hg38 FASTA and tabulate the lengths of the canonical chromosomes.

    Parameters
    ----------
    fasta_path : str, optional
        Path to the hg38 FASTA. Must already have a ``.fai`` index (pyfaidx
        builds one on first access if the file is writable). Hard-coded
        default ``data/hg38.fa``; adjust before rerunning elsewhere.

    Returns
    -------
    hg38 : pyfaidx.Fasta
        Random-access genome object; ``hg38[chrom][start:end].seq`` returns the
        sequence of a 0-based, half-open interval.
    chrom_len : dict of str to int
        Chromosome length in bp for chr1-22, chrX and chrY — deliberately not
        "all contigs", because the CRE tables only reference these. Used to
        clip windows that hang off the chromosome end before padding with N.

    Notes
    -----
    Loading the full index costs a few seconds; the object is then shared by
    all batches (pyfaidx keeps an open file handle, so do not pickle it).
    """
    hg38 = Fasta(fasta_path)
    chrom_len = {c: len(hg38[c]) for c in [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']}
    return hg38, chrom_len


# ── Ontology term mapping ────────────────────────────────────────
# AlphaGenome addresses cell types by Experimental Factor Ontology / Cell
# Ontology accession rather than by free-text name, so every cell line in the
# benchmark is mapped here once. WTC11 is the iPSC line used for the hPSC
# experiments and is deliberately aliased to the same CL term as hPSC
# (CL:0000047), which mirrors the P0_RELABEL = {'WTC11': 'hPSC'} convention
# used by the other benchmark scripts.
ONTOLOGY_TERM = {
    'K562': 'EFO:0002067',
    'HepG2': 'EFO:0001187',
    'HT29': 'EFO:0001099',
    'A375': 'CL:0002567',
    'hPSC': 'CL:0000047',
    'WTC11': 'CL:0000047',
}


def check_ontology(cell_lines):
    """Fail fast if a CRE table contains a cell line with no ontology mapping.

    Parameters
    ----------
    cell_lines : iterable of str
        Cell-line labels as they appear in the input table (e.g. the unique
        values of ``cre_df['cell_line']``).

    Returns
    -------
    None
        The function only validates and prints; the mapping itself lives in
        :data:`ONTOLOGY_TERM`.

    Raises
    ------
    ValueError
        If any label is missing from :data:`ONTOLOGY_TERM`. Raising here is
        preferable to a ``KeyError`` deep inside a batch loop, because a
        partially written HDF5 is expensive to diagnose.

    Notes
    -----
    The printed value counts are a cheap sanity check on the input table: the
    FACS dataset is expected to contain exactly K562 and hPSC (Supplementary
    Methods §5.5).
    """
    unknown = set(cell_lines) - set(ONTOLOGY_TERM.keys())
    if unknown:
        raise ValueError(f'Unknown cell lines: {unknown}')
    print('Ontology OK:', dict(pd.Series(list(cell_lines)).value_counts()))


# ── Perturbation parameters ──────────────────────────────────────
# One AlphaGenome call per sequence; the API bills and rate-limits per request,
# so batches are kept small (8) and the three strategies are spread over the
# three API keys rather than enlarging the batch.
BATCH_SIZE = 8
WINDOW_SIZE = 512        # perturbation window size (512 bp)
HALF_WINDOW = WINDOW_SIZE // 2  # 256 bp

# AlphaGenome consumes exactly 1,048,576 bp (1 Mb) of sequence. The centre of
# that window is 524,288, and the perturbed 512 bp interval is therefore
# [524,032, 524,544) in *window-local* coordinates — i.e. 256 bp on each side
# of the window centre. These are window offsets, NOT genomic coordinates: the
# genomic position of the perturbed bases is ``row.begin + PERTURB_START``.
# Center of the 1 Mb sequence is 524,288; perturbed region [524,032, 524,544)
PERTURB_CENTER = 524288
PERTURB_START = PERTURB_CENTER - HALF_WINDOW  # 524,032
PERTURB_END = PERTURB_CENTER + HALF_WINDOW    # 524,544

SEQUENCE_LENGTH_1MB = 1048576


# ── Sequence construction + prediction ───────────────────────────
def prepare_sequences_fast(batch_rows, hg38, chrom_len):
    """
    Build WT / MASK / SHUFFLE sequences for a batch.
    MASK:   replace the fixed window (PERTURB_START:PERTURB_END) with N
    SHUFFLE: randomly shuffle the window (per-sample fixed seed, reproducible)

    Parameters
    ----------
    batch_rows : pandas.DataFrame
        Rows of the CRE table for one batch. Required columns: ``chr`` (str),
        ``begin`` and ``stop`` (genomic window bounds, 0-based half-open),
        ``cell_line``, ``id``. ``begin``/``stop`` are expected to already span
        1 Mb around the CRE summit (built by the caller, see
        ``build_cre_list`` / ``build_cre_df``).
    hg38 : pyfaidx.Fasta
        Indexed reference genome.
    chrom_len : dict of str to int
        Chromosome lengths, used to clip windows that run past the chromosome
        end before padding.

    Returns
    -------
    wt_seqs : list of str
        Unperturbed 1,048,576 bp sequences, one per row.
    mask_seqs : list of str
        Same sequences with ``[524,032, 524,544)`` replaced by ``N``.
    shuffle_seqs : list of str
        Same sequences with ``[524,032, 524,544)`` shuffled in place.
    cell_line : str
        Cell line shared by the batch (the caller groups rows by cell line, so
        this is taken from the first row and assumed homogeneous).

    Notes
    -----
    * Sequences shorter than 1 Mb after clipping are *centre-padded* with ``N``
      to exactly :data:`SEQUENCE_LENGTH_1MB`. ``str.center`` puts the extra
      padding asymmetrically when the deficit is odd (the left side gets one
      fewer ``N`` than the right), so for a window clipped at only one end the
      perturbed offsets can be off by one base. This affects only CREs within
      ~0.5 Mb of a chromosome end and does not change the assay for the
      benchmark loci.
    * The shuffle uses ``np.random.seed(114514 + i)`` where ``i`` is the
      position within the batch — not a stable sample id. Batch composition
      therefore influences the masked shuffle, though the seed itself makes a
      given run reproducible.
    * The ``N`` mask is a per-base substitution only; AlphaGenome receives the
      same sequence length in all three arms, so any score difference is
      attributable to the window content alone.
    * Rows are re-iterated with ``iterrows()``, which yields views; nothing is
      written back to ``batch_rows``.
    """
    n_samples = len(batch_rows)
    wt_seqs      = [None] * n_samples
    mask_seqs    = [None] * n_samples
    shuffle_seqs = [None] * n_samples

    for i, (_, rw) in enumerate(batch_rows.iterrows()):
        chr_name = rw['chr']
        start = max(0, int(rw['begin']))
        end   = min(chrom_len[chr_name], int(rw['stop']))

        seq = hg38[chr_name][start:end].seq.upper()
        if len(seq) < SEQUENCE_LENGTH_1MB:
            seq = seq.center(SEQUENCE_LENGTH_1MB, 'N')

        wt_seqs[i] = seq

        # MASK: window -> N
        seq_list = list(seq)
        for pos in range(PERTURB_START, min(PERTURB_END, len(seq_list))):
            seq_list[pos] = 'N'
        mask_seqs[i] = ''.join(seq_list)

        # SHUFFLE: shuffle the window (fixed seed, reproducible)
        shuffle_list = list(seq)
        perturb_region = shuffle_list[PERTURB_START:PERTURB_END]
        np.random.seed(114514 + i)
        np.random.shuffle(perturb_region)
        shuffle_list[PERTURB_START:PERTURB_END] = perturb_region
        shuffle_seqs[i] = ''.join(shuffle_list)

        del seq_list, shuffle_list, perturb_region

    return wt_seqs, mask_seqs, shuffle_seqs, batch_rows.iloc[0]['cell_line']


def predict_single_model(model, sequences, ontology_term, max_workers):
    """CAGE prediction wrapper for a single model

    Runs one AlphaGenome request over a list of DNA sequences, requesting only
    the CAGE output head for a single cell-type ontology term.

    Parameters
    ----------
    model : alphagenome.models.dna_client.DnaClient
        One of the clients returned by :func:`create_models`.
    sequences : list of str
        DNA sequences of exactly 1,048,576 bp each (ACGT plus ``N``).
    ontology_term : str
        Ontology accession for the target cell type, e.g. ``'EFO:0002067'``
        for K562 (see :data:`ONTOLOGY_TERM`).
    max_workers : int
        Number of threads the client may use to parallelise the *internal*
        inference across sequences. Callers pass ``len(sequences)``.

    Returns
    -------
    list
        One prediction object per input sequence. ``result[i].cage.values`` is
        a ``(n_bins, n_tracks)`` array of CAGE signal; downstream code sums it
        over both axes (see :func:`process_batch_parallel`).

    Notes
    -----
    Only CAGE is requested. Supplementary Methods §6.2.2 describes the
    perturbation as being read out on the RNA-seq tracks, whereas the released
    AlphaGenome API path used here reads the CAGE head; §5.5 describes the CAGE
    readout. The code follows §5.5 — see the module docstring warning in
    ``run_escreen_perturbation.py``.
    """
    return model.predict_sequences(
        sequences=sequences,
        requested_outputs=[OutputType.CAGE],
        ontology_terms=[ontology_term],
        progress_bar=False,
        max_workers=max_workers
    )


def process_batch_parallel(model1, model2, model3, batch_rows, hg38, chrom_len):
    """
    Run 3-model parallel prediction for one batch.
    Returns [(sample_id, wt_cage, mask_cage, shuffle_cage, row), ...]

    Parameters
    ----------
    model1, model2, model3 : DnaClient
        Clients bound to three distinct API keys; model1 handles WT, model2
        MASK, model3 SHUFFLE.
    batch_rows : pandas.DataFrame
        One batch of CRE rows (see :func:`prepare_sequences_fast`).
    hg38 : pyfaidx.Fasta
        Indexed reference genome.
    chrom_len : dict of str to int
        Chromosome lengths.

    Returns
    -------
    list of tuple
        ``(sample_id, wt_cage, mask_cage, shuffle_cage, row)`` per sample,
        where ``sample_id`` is ``"{id}@{cell_line}"``, each ``*_cage`` is a
        ``float16`` vector of per-bin CAGE signal (CAGE values summed across
        tracks), and ``row`` is the original ``pandas.Series``.

    Notes
    -----
    * The three ``executor.submit`` calls are plain threads: the work is I/O
      bound (HTTP to the hosted API), so the GIL is not a bottleneck, and this
      keeps the three requests in flight simultaneously.
    * CAGE tracks are summed with ``.sum(axis=1)``, i.e. collapsed over tracks
      but *kept per output bin*. The perturbation score is a later reduction
      over bins (``sum(WT - ALT)``) done by the analysis scripts, not here.
    * Values are cast to ``float16`` purely to halve the HDF5 footprint; the
      losses (~1e-3 relative) are irrelevant next to the WT-vs-perturbed
      differences being scored.
    """
    wt_seqs, mask_seqs, shuffle_seqs, cell_line = prepare_sequences_fast(batch_rows, hg38, chrom_len)
    ontology_term = ONTOLOGY_TERM[cell_line]
    n_samples = len(batch_rows)

    with ThreadPoolExecutor(max_workers=3) as executor:
        f_wt = executor.submit(predict_single_model, model1, wt_seqs, ontology_term, n_samples)
        f_mask = executor.submit(predict_single_model, model2, mask_seqs, ontology_term, n_samples)
        f_shuffle = executor.submit(predict_single_model, model3, shuffle_seqs, ontology_term, n_samples)
        wt_results = f_wt.result()
        mask_results = f_mask.result()
        shuffle_results = f_shuffle.result()

    results = []
    for i, (_, rw) in enumerate(batch_rows.iterrows()):
        sample_id = f"{rw.id}@{rw.cell_line}"
        wt_cage      = wt_results[i].cage.values.sum(axis=1).astype('float16')
        mask_cage    = mask_results[i].cage.values.sum(axis=1).astype('float16')
        shuffle_cage = shuffle_results[i].cage.values.sum(axis=1).astype('float16')
        results.append((sample_id, wt_cage, mask_cage, shuffle_cage, rw))

    del wt_seqs, mask_seqs, shuffle_seqs, wt_results, mask_results, shuffle_results
    return results


# ── Checkpoint / resume management ───────────────────────────────
class CheckpointManager:
    """Checkpoint management for resume (pickle: processed/failed index sets)

    Tracks which row indices of the input CRE table have already been written to
    the output HDF5, so that a long API run can be interrupted and resumed.

    Parameters
    ----------
    checkpoint_file : str, optional
        Path of the pickle used to persist the two index sets. The file is
        removed by :meth:`cleanup` once every sample is done.

    Attributes
    ----------
    processed : set of int
        Indices already saved to HDF5.
    failed : set of int
        Indices whose batch raised; retried first by
        :meth:`get_pending_indices`.

    Notes
    -----
    Indices are positions into the CRE table (``range(len(cre_df))``), so the
    checkpoint is only valid together with the exact input table it was created
    from. The HDF5 file remains the source of truth: the runner re-syncs
    ``processed`` from the groups already present, so a stale or deleted pickle
    only costs re-checking, not duplicate work.

    Saves are throttled: :meth:`mark_processed` flushes every 100 samples, while
    :meth:`mark_failed` flushes immediately (failures are rare and worth
    persisting).
    """

    def __init__(self, checkpoint_file='checkpoint.pkl'):
        self.checkpoint_file = checkpoint_file
        self.processed = set()
        self.failed = set()
        self.load()

    def load(self):
        """Read the pickle if present, silently starting fresh if it is unreadable."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                    self.processed = data.get('processed', set())
                    self.failed = data.get('failed', set())
                print(f"Loaded checkpoint: {len(self.processed)} processed, "
                      f"{len(self.failed)} failed")
            except Exception:
                print("Failed to load checkpoint, starting fresh")

    def save(self):
        """Serialise both index sets to :attr:`checkpoint_file`."""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({
                'processed': self.processed,
                'failed': self.failed
            }, f)

    def mark_processed(self, idx):
        """Record one successfully written sample and flush every 100 samples.

        Parameters
        ----------
        idx : int
            Row index into the CRE table.

        Notes
        -----
        A sample that previously failed is removed from ``failed`` here, so a
        successful retry stops being scheduled.
        """
        self.processed.add(idx)
        self.failed.discard(idx)
        # save every 100 samples
        if len(self.processed) % 100 == 0:
            self.save()

    def mark_failed(self, idx):
        """Record one sample whose batch raised, and flush immediately.

        Parameters
        ----------
        idx : int
            Row index into the CRE table.
        """
        self.failed.add(idx)
        self.save()

    def get_pending_indices(self, total_indices):
        """Pending indices (previously failed ones first)

        Parameters
        ----------
        total_indices : int
            Size of the input CRE table, i.e. ``len(cre_df)``.

        Returns
        -------
        list of int
            Indices not yet processed, with the previously failed subset moved
            to the front so that a resumed run retries the problematic samples
            while the GPU/API quota is still warm. Order within each subset is
            ``set`` order and therefore not reproducible.
        """
        all_indices = set(range(total_indices))
        pending = all_indices - self.processed
        failed_pending = self.failed & pending
        other_pending = pending - self.failed
        return list(failed_pending) + list(other_pending)

    def clear_failed(self):
        """Forget all failed indices (useful before a manual re-run of a bad batch)."""
        self.failed.clear()
        self.save()

    def cleanup(self, total_samples):
        """Remove the checkpoint file when everything is done

        Parameters
        ----------
        total_samples : int
            Expected number of samples; the file is only deleted when
            ``len(processed) == total_samples`` so that an interrupted run keeps
            its progress record.

        Notes
        -----
        Deletion can fail on Windows if the HDF5 file handle is still open or
        the pickle is locked by another process; there is no retry here.
        """
        if len(self.processed) == total_samples and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print('Checkpoint removed')


# ── String perturbation utilities (windowed operations) ──────────
# Generic, in-memory equivalents of the operations applied by
# prepare_sequences_fast. They operate on an arbitrary string (normally a 1 Mb
# DNA sequence) and on window offsets relative to that string. All bounds are
# clamped, so callers may pass windows that run past either end. They are kept
# for ad-hoc/interactive use; the batched runner path uses the faster list-based
# code in prepare_sequences_fast instead.
def mask_in_window(original_str: str, start: int, end: int) -> str:
    """Replace all characters in the given window with 'N'.

    Parameters
    ----------
    original_str : str
        Sequence to modify.
    start, end : int
        0-based, half-open window offsets into ``original_str``; clamped to the
        valid range.

    Returns
    -------
    str
        A new string of the same length. The input is never modified in place,
        and an empty input is returned unchanged.
    """
    if not original_str:
        return original_str
    chars = list(original_str)
    actual_start = max(0, start)
    actual_end = min(len(original_str), end)
    for i in range(actual_start, actual_end):
        chars[i] = 'N'
    return ''.join(chars)


def replace_in_window(original_str: str, start: int, end: int,
                      target_char: str, replacement_char: str) -> str:
    """Replace target characters with replacement characters in the window.

    Parameters
    ----------
    original_str : str
        Sequence to modify.
    start, end : int
        0-based, half-open window offsets; clamped to the string bounds.
    target_char : str
        Single character to look for, e.g. ``'N'``.
    replacement_char : str
        Single character to substitute, e.g. ``'A'``.

    Returns
    -------
    str
        New string of the same length.

    Raises
    ------
    ValueError
        If either character argument is not exactly one character long.

    Notes
    -----
    Only exact character matches inside the window are touched; positions that
    fall outside ``[start, end)`` are copied through untouched.
    """
    if not original_str:
        return original_str
    if len(target_char) != 1 or len(replacement_char) != 1:
        raise ValueError("target_char and replacement_char must be single characters")
    chars = list(original_str)
    actual_start = max(0, start)
    actual_end = min(len(original_str), end)
    for i in range(actual_start, actual_end):
        if chars[i] == target_char:
            chars[i] = replacement_char
    return ''.join(chars)


def shuffle_in_window(original_str: str, start: int, end: int, seed=114514) -> str:
    """Randomly shuffle the characters in the given window.

    Parameters
    ----------
    original_str : str
        Sequence to modify.
    start, end : int
        0-based, half-open window offsets; clamped to the string bounds.
    seed : int, optional
        Seed for :func:`random.seed`. Default ``114514`` is the project-wide
        seed used by the shuffling perturbation, so the same input window always
        yields the same shuffled sequence.

    Returns
    -------
    str
        New string of the same length with the window's characters permuted.

    Notes
    -----
    This permutes bases within the window only — it preserves the window's base
    composition, which is what makes the shuffling control in Supplementary
    Methods §6.2.2 different from N-masking: the mask removes all information,
    the shuffle preserves mononucleotide content (and hence most simple
    k-mer-composition effects) while destroying motif order.

    A window of length <= 1 (or an inverted/empty interval) is returned
    unchanged, since shuffling it is a no-op.

    Uses Python's global :mod:`random` state; callers in a multithreaded context
    would need to serialise access.
    """
    if not original_str:
        return original_str
    actual_start = max(0, start)
    actual_end = min(len(original_str), end)
    if actual_start >= actual_end or (actual_end - actual_start) <= 1:
        return original_str
    random.seed(seed)
    chars = list(original_str)
    window_chars = chars[actual_start:actual_end]
    random.shuffle(window_chars)
    chars[actual_start:actual_end] = window_chars
    return ''.join(chars)


# ── Evaluation metric functions ──────────────────────────────────
# Metrics follow Supplementary Methods §7.1 (AUROC / AUPRC for the CRISPR
# benchmark) and the m2a composite used for checkpoint selection in §5.5.
def z_norm(header: dict, values: np.ndarray) -> np.ndarray:
    """Z-normalise per-bin signal using bigWig summary statistics.

    Parameters
    ----------
    header : dict
        Summary statistics for the track: ``"sumData"``, ``"sumSquared"`` and
        ``"nBasesCovered"`` (the fields returned by
        ``pyBigWig.stats(..., type="summary")``).
    values : numpy.ndarray
        Raw signal values to normalise, any shape.

    Returns
    -------
    numpy.ndarray
        ``(values - mean) / std`` computed from the header, same shape as
        ``values``.

    Raises
    ------
    ValueError
        If a required header key is missing, ``nBasesCovered == 0``, or the
        standard deviation is exactly 0 (constant track).

    Notes
    -----
    The variance is clamped at 0 with ``max(...)`` before the square root: for a
    constant track ``mean_sq - mean**2`` can be a tiny negative number in
    floating point, and ``np.sqrt`` of that is ``nan``.

    Not used by the current AlphaGenome scoring path (the perturbation score is
    a raw difference of CAGE sums); it is a shared utility for track-normalised
    comparisons elsewhere in the analysis notebooks.
    """
    required_keys = {"sumData", "sumSquared", "nBasesCovered"}
    if not required_keys.issubset(header.keys()):
        raise ValueError(f"Header must contain keys: {required_keys}")
    n = header["nBasesCovered"]
    if n == 0:
        raise ValueError("nBasesCovered is 0, cannot compute mean/std")
    mean = header["sumData"] / n
    mean_sq = header["sumSquared"] / n
    variance = max(mean_sq - mean**2, 0.0)
    std = np.sqrt(variance)
    if std == 0:
        raise ValueError("Std is 0, cannot perform Z normalization")
    return (values - mean) / std


def Logistic_norm(x, px=[0, 2.0, 4.0, 6.0], py=[0.0, 0.5, 0.8, 1.0]):
    """Map raw tRRA activity scores onto the [0, 1] label range.

    This is the nonlinear label transform of Supplementary Methods §2.3.2
    (``def Logistic_norm(x, px=[0, 2.0, 4.0, 6.0], py=[0.0, 0.5, 0.8, 1.0])``).

    Parameters
    ----------
    x : array_like
        Raw tRRA values (``-log10`` MAGeCK RRA). Values outside ``px`` are
        clamped by the interpolator.
    px : list of float, optional
        Abscissae of the control points: tRRA 0, 2, 4 and 6 — i.e. RRA 1, 1e-2,
        1e-4 and 1e-6, matching the tRRA > 2 threshold that defines a functional
        CRE (§1.4).
    py : list of float, optional
        Ordinates at those control points: 0.0, 0.5, 0.8 and 1.0.

    Returns
    -------
    numpy.ndarray
        Activity labels in approximately [0, 1], same shape as ``x``.

    Notes
    -----
    * PCHIP (monotone piecewise cubic Hermite) interpolation is used rather than
      a logistic function despite the name: it interpolates the control points
      exactly and cannot overshoot, so the output never leaves [0, 1].
    * Not to be confused with the sigmoid-based *sigmoid normalisation* of
      gRNA LFC values described in Supplementary Methods §1.3; that one happens
      upstream, on gRNA LFCs, not here on CRE-level tRRA.
    * ``Logistic_norm`` is applied to labels, not predictions; models are trained
      with a smooth-L1 loss against these values plus an auxiliary BCE term
      (§2.3.2, §5.5).
    """
    x = np.asarray(x)
    return PchipInterpolator(px, py)(x)


def build_label_boost_weights(labels, target_pos_mass=0.55):
    """Per-sample weights that give the positive class a fixed share of the total mass.

    Implements the weighting side of the label-boosted sampler used for the
    eScreen benchmark training (Supplementary Methods §2.3.4 and §5.5, "target
    positive class mass fraction of 0.55", temperature 0.45 applied separately
    by the sampler).

    Parameters
    ----------
    labels : array_like
        Binary labels (0/1) or scores thresholded at 0.5.
    target_pos_mass : float, optional
        Desired total weight of the positive class, default 0.55.

    Returns
    -------
    numpy.ndarray
        Weights of shape ``(n,)`` summing to 1, with all positives sharing
        ``target_pos_mass`` and all negatives sharing ``1 - target_pos_mass``.

    Notes
    -----
    * This rewrites class prior only; it does not resample. The sampler used at
      training time dirties this further with a temperature exponent.
    * Epsilon guards: counts are floored at 1 so an all-positive or
      all-negative batch cannot divide by zero, and the final normalisation uses
      ``max(w.sum(), 1e-12)`` for the degenerate all-zero case.
    """
    y = np.asarray(labels, dtype=float)
    pos = y > 0.5
    n_pos = max(pos.sum(), 1)
    n_neg = max((~pos).sum(), 1)
    w = np.where(pos, target_pos_mass / n_pos, (1.0 - target_pos_mass) / n_neg)
    return w / max(w.sum(), 1e-12)


def median_auroc(pred, y_true, groups):
    """Median of per-group AUROC, skipping groups that cannot be scored.

    Parameters
    ----------
    pred : array_like
        Continuous scores (higher = more likely positive), shape ``(n,)``.
    y_true : array_like
        Binary labels, shape ``(n,)``.
    groups : array_like
        Group label per sample, normally the cell line, shape ``(n,)``.

    Returns
    -------
    float
        Median AUROC across groups, or ``nan`` if no group was scoreable.

    Notes
    -----
    A group is skipped when it has fewer than 2 samples *or* only one distinct
    label — ``roc_auc_score`` raises in both cases, and silently omitting the
    group is the convention used for the per-cell-line benchmark tables
    (Supplementary Methods §6.1 reports per-cell-line AUROC).

    The median (rather than the mean) is used because the number of CREs differs
    substantially between cell lines; it is also the first term of the m2a
    composite score.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups)
    scores = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2 or len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], pred[idx]))
    if not scores:
        return float('nan')
    return float(np.median(scores))


def median_auprc(pred, y_true, groups):
    """Median of per-group AUPRC (area under the precision-recall curve).

    Parameters
    ----------
    pred : array_like
        Continuous scores, shape ``(n,)``.
    y_true : array_like
        Binary labels, shape ``(n,)``.
    groups : array_like
        Group label per sample (cell line), shape ``(n,)``.

    Returns
    -------
    float
        Median AUPRC across groups, or ``nan`` if there were no groups.

    Notes
    -----
    * Unlike :func:`median_auroc`, groups are only skipped when they hold fewer
      than 2 samples: ``precision_recall_curve`` tolerates a single-class group,
      and dropping such groups would bias the median.
    * ``auc(rec, prec)`` is used rather than ``average_precision_score`` so that
      the number matches the trapezoidal AUPRC reported in the manuscript
      tables; ``sklearn.metrics.auc`` auto-detects that recall is decreasing and
      flips the sign, so the integral comes out positive.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups)
    scores = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true[idx], pred[idx])
        scores.append(auc(rec, prec))
    if not scores:
        return float('nan')
    return float(np.median(scores))


def m2a_combo_score(pred, y_true, groups):
    """m2a_combo = 0.7 x median_AUROC + 0.3 x median_AUPRC

    The composite model-selection score of Supplementary Methods §5.5: the best
    checkpoint is chosen by the m2a composite score rather than by a single
    metric, so a model cannot win by trading AUROC against AUPRC.

    Parameters
    ----------
    pred : array_like
        Continuous scores, shape ``(n,)``.
    y_true : array_like
        Binary labels, shape ``(n,)``.
    groups : array_like
        Cell-line label per sample, shape ``(n,)``.

    Returns
    -------
    float
        The weighted composite, or ``nan`` if the AUROC median is ``nan``.

    Notes
    -----
    * The 0.7/0.3 weighting is asymmetric on purpose: AUROC is the primary
      benchmark metric in Supplementary Methods §6.1, AUPRC is the tie-breaker
      that penalises poor ranking of the minority (active) class.
    * This returns a **scalar**. Callers that unpack it into two values (see
      ``analyze_facs_scores.py``, which does
      ``combo, results = m2a_combo_score(...)``) are broken against this
      signature; that script predates the change to a scalar return.
    """
    ma = median_auroc(pred, y_true, groups)
    mp = median_auprc(pred, y_true, groups)
    return 0.7 * ma + 0.3 * mp


def per_cell_line_metrics(pred, y_true, groups):
    """Compute AUROC and AUPRC separately for every group (cell line).

    Parameters
    ----------
    pred : array_like
        Continuous scores, shape ``(n,)``.
    y_true : array_like
        Binary labels, shape ``(n,)``.
    groups : array_like
        Group label per sample, shape ``(n,)``.

    Returns
    -------
    list of dict
        One entry per scoreable group with keys ``'cell_line'``, ``'AUROC'``,
        ``'AUPRC'``, ``'n_pos'``, ``'n_neg'``. Groups containing only one class
        are omitted, since both metrics are undefined there.

    Notes
    -----
    ``pred`` is expected to be already monotone-transformed (e.g. passed through
    a sigmoid); both metrics are rank based, so the transform does not matter
    for the values, but it keeps the API consistent with the scripts that feed
    probabilities.

    The returned list is consumed by :func:`print_metrics` and by the
    checkpoint-selection loops in the training scripts.
    """
    results = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        yt = y_true[idx]
        pp = pred[idx]
        n_pos = int(yt.sum())
        n_neg = int((1 - yt).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        auroc = roc_auc_score(yt, pp)
        prec, rec, _ = precision_recall_curve(yt, pp)
        auprc = auc(rec, prec)
        results.append({'cell_line': g, 'AUROC': auroc, 'AUPRC': auprc,
                        'n_pos': n_pos, 'n_neg': n_neg})
    return results


def print_metrics(results, label):
    """Pretty-print a per-cell-line metric table and return the two means.

    Parameters
    ----------
    results : list of dict
        Output of :func:`per_cell_line_metrics`.
    label : str
        Heading printed above the table, e.g. ``'MASK strategy'``.

    Returns
    -------
    mean_auroc : float
        Unweighted mean AUROC over the rows of ``results``.
    mean_auprc : float
        Unweighted mean AUPRC over the rows of ``results``.

    Notes
    -----
    These are *macro* means over cell lines, not micro/pooled values: a cell line
    with many more test CREs does not dominate. The manuscript reports
    per-cell-line AUROC/AUPRC (§6.1) and medians via :func:`m2a_combo_score`;
    the arithmetic mean printed here is only a run-time summary.
    """
    print(f'\n=== {label} ===')
    print(f'{"Cell Line":<12} {"n_pos":<8} {"n_neg":<8} {"AUROC":<8} {"AUPRC":<8}')
    print('-' * 44)
    for r in results:
        print(f'{r["cell_line"]:<12} {r["n_pos"]:<8} {r["n_neg"]:<8} '
              f'{r["AUROC"]:<8.4f} {r["AUPRC"]:<8.4f}')
    print('-' * 44)
    mean_auroc = np.mean([r['AUROC'] for r in results])
    mean_auprc = np.mean([r['AUPRC'] for r in results])
    print(f'{"MEAN":<12} {"":<8} {"":<8} {mean_auroc:<8.4f} {mean_auprc:<8.4f}')
    return mean_auroc, mean_auprc
