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
    """CAGE prediction wrapper for a single model"""
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
    """Checkpoint management for resume (pickle: processed/failed index sets)"""

    def __init__(self, checkpoint_file='checkpoint.pkl'):
        self.checkpoint_file = checkpoint_file
        self.processed = set()
        self.failed = set()
        self.load()

    def load(self):
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
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({
                'processed': self.processed,
                'failed': self.failed
            }, f)

    def mark_processed(self, idx):
        self.processed.add(idx)
        self.failed.discard(idx)
        # save every 100 samples
        if len(self.processed) % 100 == 0:
            self.save()

    def mark_failed(self, idx):
        self.failed.add(idx)
        self.save()

    def get_pending_indices(self, total_indices):
        """Pending indices (previously failed ones first)"""
        all_indices = set(range(total_indices))
        pending = all_indices - self.processed
        failed_pending = self.failed & pending
        other_pending = pending - self.failed
        return list(failed_pending) + list(other_pending)

    def clear_failed(self):
        self.failed.clear()
        self.save()

    def cleanup(self, total_samples):
        """Remove the checkpoint file when everything is done"""
        if len(self.processed) == total_samples and os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print('Checkpoint removed')


# ── String perturbation utilities (windowed operations) ──────────
def mask_in_window(original_str: str, start: int, end: int) -> str:
    """Replace all characters in the given window with 'N'."""
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
    """Replace target characters with replacement characters in the window."""
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
    """Randomly shuffle the characters in the given window."""
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
def z_norm(header: dict, values: np.ndarray) -> np.ndarray:
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
    x = np.asarray(x)
    return PchipInterpolator(px, py)(x)


def build_label_boost_weights(labels, target_pos_mass=0.55):
    y = np.asarray(labels, dtype=float)
    pos = y > 0.5
    n_pos = max(pos.sum(), 1)
    n_neg = max((~pos).sum(), 1)
    w = np.where(pos, target_pos_mass / n_pos, (1.0 - target_pos_mass) / n_neg)
    return w / max(w.sum(), 1e-12)


def median_auroc(pred, y_true, groups):
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
    """m2a_combo = 0.7 x median_AUROC + 0.3 x median_AUPRC"""
    ma = median_auroc(pred, y_true, groups)
    mp = median_auprc(pred, y_true, groups)
    return 0.7 * ma + 0.3 * mp


def per_cell_line_metrics(pred, y_true, groups):
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
