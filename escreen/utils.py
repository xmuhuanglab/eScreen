"""Shared utilities for the eScreen training and evaluation pipeline.

This module is a supporting toolbox for the genome-wide eScreen model. It
contains no network layers, no sequence handling and no file I/O: every helper
below operates either on plain NumPy arrays or on the ``dict``-style batches
produced by the training loop. Four functional groups are provided.

1. Class-balanced sampling (:class:`boost_sampler`) used to oversample
   low/high-activity CRE-cell line pairs during training. This is the
   temperature-controlled sampler of Supplementary Methods §2.3.4; the
   manuscript reproduces the same pseudocode.
2. Ranking and correlation metrics (AUROC, AUPRC, per-group Spearman,
   precision@k, recall@k). These implement the metric set of Supplementary
   Methods §7.1 (AUROC, AUPRC, top-k precision; Pearson correlation is used
   there for the regression tasks -- see the note on Spearman below).
3. Metric factories that wrap such a metric into a single scalar
   ``f(model, batch, device) -> float`` score, so that the training loop can
   use any of them interchangeably as an early-stopping / checkpoint-selection
   criterion.
4. Success-gate helpers that turn reported metrics into boolean pass/fail
   switches for the internal training phases ``N1``-``N6``.

Notes
-----
* The manuscript defines the regression-target mapping (``Logistic_norm``, a
  PCHIP interpolation of the MAGeCK RRA value, Supplementary Methods §2.3.2)
  and the weighted smooth-L1 + BCE training loss (§2.3.2). **Neither is
  implemented in this file**; only the §2.3.4 sampler is. The module name
  ``utils`` is broader than its contents.
* The phase labels ``N1``-``N6``, the "golden intersection" evaluation and
  every numeric gate threshold are internal experiment-management conventions.
  They are *not* described in the manuscript, and no manuscript section is
  cited for them.
* The per-group correlation metric here is Spearman (rank) correlation, whereas
  Supplementary Methods §7.1 specifies Pearson correlation for the regression
  tasks. The two are not interchangeable; the Spearman variant is what this
  file provides and what the training loop consumes.
* All metric helpers return ``float("nan")`` instead of raising when a metric
  is undefined (single-class target, group too small, constant predictions).
  Callers detect this with the ``v == v`` idiom, which fails for ``nan``; that
  convention is used throughout the module and must be preserved by callers.
* ``torch``, ``torch.nn`` and ``torch.nn.functional`` are imported but not
  referenced anywhere in this module; all helpers are NumPy / scikit-learn /
  SciPy only.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import re
from typing import Any, Sequence
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

class boost_sampler():
    """Temperature-controlled class-balanced sampler for imbalanced CRE labels.

    Implements the sampler of Supplementary Methods §2.3.4. Training indices are
    drawn *with replacement* from a per-sample distribution in which rare
    classes are up-weighted by inverse (smoothed) frequency and the resulting
    weights are softened by a temperature exponent:

    ``smoothed_counts = counts + smoothing``
    ``class_weights_raw = n_total / smoothed_counts``
    ``sample_weight = class_weights_raw[class] ** temperature``
    ``prob = sample_weight / sum(sample_weight)``

    The ``temperature=0.7`` default in the signature is the paper's default
    argument; the eScreen training runs used ``T = 0.45``
    (Supplementary Methods §2.3.4). Lower temperature gives a flatter
    distribution, i.e. less aggressive repetition of rare samples.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Class label of every training sample, typically the binarised CRE
        activity label. Labels are used as dictionary keys, so they must be
        hashable scalars.
    smoothing : float, default 1.0
        Laplace-style additive smoothing applied to the class counts.
    temperature : float, default 0.7
        Exponent applied to the raw inverse-frequency weights: ``1.0`` is fully
        inverse-frequency, ``< 1`` flattens towards uniform, ``0`` is uniform.
    seed : int, default 114514
        Seed of the private ``numpy.random.Generator``.

    Attributes
    ----------
    prob_dist : numpy.ndarray of shape (n_samples,)
        Per-sample sampling probability, normalised to sum to 1.

    Notes
    -----
    * The sampler does not draw a set of *unique* indices; duplicates are the
      point (see :meth:`get`).
    * Only class frequencies enter the weights. No per-sample confidence or
      loss value is used, so this is a purely label-driven sampler.
    * The distribution is computed once from a fixed label vector, so the
      sampler assumes a static training set.
    """
    
    def __init__(self, labels, smoothing=1.0, temperature=0.7, seed=114514):
        """Initialize the sampler and precompute the sampling distribution.

        Parameters
        ----------
        labels : array-like of shape (n_samples,)
            Training-set labels, typically the binarised CRE activity label
            (0 = low activity, 1 = high activity). Stored as a NumPy array; the
            values serve as dictionary keys, so they must be hashable scalars.
        smoothing : float, default 1.0
            Laplace-style additive smoothing on the per-class counts, applied
            before the inverse-frequency weights are formed.
        temperature : float, default 0.7
            Exponent on the raw inverse-frequency weights: ``1.0`` is fully
            inverse-frequency, values below 1 flatten the distribution towards
            uniform, and ``0`` gives uniform sampling. Supplementary Methods
            §2.3.4 states that eScreen training used ``T = 0.45``, i.e. flatter
            than this default.
        seed : int, default 114514
            Seed of the private ``numpy.random.Generator``. Reproducibility goes
            through that generator only; the global NumPy random state is left
            untouched.

        Returns
        -------
        None

        Notes
        -----
        The whole distribution is built once here rather than per epoch, so every
        epoch draws from exactly the same reproducible distribution. See
        :meth:`_precompute_probabilities` for the formula.
        """
        self.labels = np.array(labels)
        self.smoothing = smoothing
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)  # use the safer numpy random generator
        
        # precompute sampling probabilities to avoid recomputing every epoch
        self.prob_dist = self._precompute_probabilities()

    def _precompute_probabilities(self):
        """Build the normalised per-sample sampling distribution.

        This is a literal implementation of the pseudocode in
        Supplementary Methods §2.3.4; the class docstring states the formula.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            Sampling probabilities summing to 1, aligned with ``self.labels``.

        Notes
        -----
        * ``smoothing`` is added to the counts *before* the ratio is formed.
          Because ``np.unique`` only returns observed classes, every count is
          already ``>= 1``, so the addition is not needed to avoid a division by
          zero. Its real effect is to damp the ratio: two classes with counts
          90/10 give raw weights 9.9/0.9 with ``smoothing=1`` instead of
          10.0/1.0, which keeps a very rare class from being repeated
          excessively.
        * The temperature is applied to the raw weights; that is equivalent to
          applying it after the ``n_total`` factor, which is a constant and
          cancels in the final normalisation.
        * The per-sample mapping goes through a Python dict, so this is
          O(n_samples) Python-level work done exactly once.
        """
        n_total = len(self.labels)
        classes, counts = np.unique(self.labels, return_counts=True)
        
        # 1. compute smoothed weights: w = N / (count + alpha)
        smoothed_counts = counts + self.smoothing
        class_weights_raw = n_total / smoothed_counts
        
        # 2. temperature scaling: w = w^T
        # T < 1.0 shrinks the gap between high and low weights
        class_weights_map = {
            cls: np.power(class_weights_raw[i], self.temperature) 
            for i, cls in enumerate(classes)
        }
        
        # 3. map and normalize
        sample_weights = np.array([class_weights_map[l] for l in self.labels])
        return sample_weights / np.sum(sample_weights)

    def get(self, num_samples=None):
        """Draw the sample indices for one epoch (with replacement).

        Parameters
        ----------
        num_samples : int, optional
            Number of indices to draw. Defaults to ``len(self.labels)``, which
            keeps the number of optimisation steps per epoch identical to
            unweighted training.

        Returns
        -------
        indices : numpy.ndarray of shape (num_samples,) and dtype int64
            Row indices into the training set, drawn **with replacement**.
            Duplicates are expected: that repetition is the oversampling
            mechanism itself.

        Notes
        -----
        Because sampling is with replacement, the expected number of *distinct*
        samples in one epoch is lower than ``num_samples`` (about
        ``1 - 1/e`` of the set under a flat distribution). The per-sample
        probability is fixed for the whole run, so class balance is corrected on
        average over an epoch rather than exactly within every batch.
        """
        n_total = len(self.labels)
        size = num_samples if num_samples is not None else n_total
        
        # sample with replacement (Replacement=True is the core of handling imbalance)
        indices = self.rng.choice(
            np.arange(n_total), 
            size=size, 
            replace=True, 
            p=self.prob_dist
        )
        return indices


def auroc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve that degrades to ``nan`` instead of raising.

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        Binary ground truth, cast to ``int``.
    y_score : numpy.ndarray of shape (n_samples,)
        Continuous model score, higher meaning more likely positive.

    Returns
    -------
    float
        AUROC in [0, 1], or ``nan`` when ``y_true`` holds a single class.

    Notes
    -----
    * AUROC is undefined for a single-class target. Depending on the installed
      scikit-learn version this either raises ``ValueError`` or returns ``nan``;
      the explicit guard makes the behaviour version-independent and lets the
      caller skip the value with the module-wide ``v == v`` test.
    * No epsilon term is needed: AUROC is a rank statistic and is invariant to
      any monotone rescaling of the scores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        Binary ground truth, cast to ``int``.
    y_score : numpy.ndarray of shape (n_samples,)
        Continuous model score.

    Returns
    -------
    float
        Average precision; ``nan`` only for an empty input.

    Notes
    -----
    Unlike :func:`auroc_safe` this needs no single-class guard, because
    scikit-learn defines the value for degenerate input: all-negative returns
    ``0.0`` and all-positive returns ``1.0``. Those values are technically
    valid but carry no information, so an all-positive cell line will pull a
    macro/median AUPRC upwards. Callers that compare AUPRC across sets should
    also look at ``label_positive_rate`` (reported by
    :func:`ranking_metrics_bundle`) since AUPRC's no-skill baseline equals that
    rate.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    return float(average_precision_score(y_true, y_score))


def spearman_group(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    """Mean within-group Spearman correlation, averaged over groups.

    For every group (normally a cell line) the Spearman rank correlation between
    prediction and regression target is computed, and the finite values are
    averaged. This measures whether the model orders CREs correctly *within* a
    cell line, which is insensitive to per-cell-line offsets in the predicted
    activity scale.

    Parameters
    ----------
    pred : array-like of shape (n_samples,)
        Predicted regression values.
    y : array-like of shape (n_samples,)
        True regression targets (logistic-normalised activity, in [0, 1]).
    groups : array-like of shape (n_samples,)
        Group label per sample.

    Returns
    -------
    float
        Unweighted mean of the per-group coefficients, or ``nan`` when no group
        qualifies.

    Notes
    -----
    * A group is skipped when it has 10 or fewer samples (the test is
      ``len(idx) <= 10``, so 11 is the minimum), because a rank correlation on a
      handful of points is dominated by noise.
    * Constant predictions or a constant target make ``spearmanr`` return
      ``nan``; those groups are dropped with ``np.isfinite`` rather than
      poisoning the mean.
    * The coefficients are averaged directly, **not** Fisher-z transformed.
      That is a deliberate simplification: it keeps the score on the natural
      [-1, 1] scale, at the cost of the small bias that Fisher averaging would
      remove.
    * Spearman is used here, whereas Supplementary Methods §7.1 specifies
      Pearson correlation for the regression tasks. The two answer different
      questions (monotone association vs. linear agreement).
    """
    vals = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 10:
            continue
        r, _ = spearmanr(pred[idx], y[idx])
        if np.isfinite(r):
            vals.append(float(r))
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _apply_group_exclude(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    exclude_groups: set[str] | frozenset[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop whole groups from a metric computation.

    Parameters
    ----------
    y_true, y_score : numpy.ndarray of shape (n_samples,)
        Binary labels and continuous model scores.
    groups : numpy.ndarray of shape (n_samples,)
        Group assignment, normally the cell line of each sample.
    exclude_groups : set of str or None
        Groups to remove, e.g. a cell line that must not contribute to a given
        evaluation. ``None`` or an empty collection disables filtering.

    Returns
    -------
    y_true, y_score : numpy.ndarray
        The filtered arrays. When nothing is excluded the input objects are
        returned unchanged (no copy is made).
    groups : numpy.ndarray of str
        Group identifiers, always converted to ``str`` so that integer codes and
        NumPy string scalars compare consistently against the exclusion set.

    Notes
    -----
    The filter is one boolean mask applied to all three arrays, so the inputs
    must be aligned row by row. Excluding whole groups rather than individual
    samples is what keeps a macro/median statistic from being dominated by a
    group the caller cannot legitimately evaluate.
    """
    if not exclude_groups:
        return y_true, y_score, groups
    ex = {str(x) for x in exclude_groups}
    g = np.asarray(groups).astype(str)
    keep = ~np.isin(g, list(ex))
    return np.asarray(y_true)[keep], np.asarray(y_score)[keep], g[keep]


def macro_auroc_per_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
) -> float:
    """Macro-averaged AUROC over groups; every group weighs the same.

    The macro average is the unweighted mean of the per-group AUROCs, so a small
    cell line contributes as much as a large one. This corresponds to reporting
    AUROC per cell line as in Supplementary Methods §7.1; the manuscript does
    not state an aggregation rule, so macro-averaging is an implementation
    choice.

    Parameters
    ----------
    y_true, y_score : array-like of shape (n_samples,)
        Binary labels and continuous model scores.
    groups : array-like of shape (n_samples,)
        Group label per sample.
    exclude_groups : set of str, optional keyword-only
        Groups dropped before averaging; see :func:`_apply_group_exclude`.

    Returns
    -------
    float
        Mean per-group AUROC, or ``nan`` when no group is eligible.

    Notes
    -----
    A group is skipped when it has fewer than 2 samples or only one class;
    including it would either raise inside scikit-learn or contribute a
    meaningless value.
    """
    y_true, y_score, groups = _apply_group_exclude(y_true, y_score, groups, exclude_groups)
    scores = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2 or len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def macro_auprc_per_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
) -> float:
    """Macro-averaged AUPRC over groups; every group weighs the same.

    Parameters
    ----------
    y_true, y_score : array-like of shape (n_samples,)
        Binary labels and continuous model scores.
    groups : array-like of shape (n_samples,)
        Group label per sample.
    exclude_groups : set of str, optional keyword-only
        Groups dropped before averaging.

    Returns
    -------
    float
        Mean per-group average precision, or ``nan`` when no group is eligible.

    Notes
    -----
    Only the sample count (``>= 2``) is checked, not the class balance: a
    single-class group still contributes, as ``0.0`` when all-negative and
    ``1.0`` when all-positive. On a strongly imbalanced benchmark the
    all-positive case can inflate this average noticeably, so read it together
    with the per-group values from :func:`per_group_auroc_auprc`.
    """
    y_true, y_score, groups = _apply_group_exclude(y_true, y_score, groups, exclude_groups)
    scores = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def median_auroc_per_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
) -> float:
    """Median AUROC over groups; robust to one badly predicted cell line.

    Parameters
    ----------
    y_true, y_score : array-like of shape (n_samples,)
        Binary labels and continuous model scores.
    groups : array-like of shape (n_samples,)
        Group label per sample.
    exclude_groups : set of str, optional keyword-only
        Groups dropped before the median.

    Returns
    -------
    float
        Median per-group AUROC, or ``nan`` when no group is eligible.

    Notes
    -----
    Eligibility matches :func:`macro_auroc_per_group` (at least 2 samples and
    both classes present). The median is preferred over the mean when a single
    cell line has very few positives, in which case its AUROC can be close to a
    coin flip regardless of model quality.
    """
    y_true, y_score, groups = _apply_group_exclude(y_true, y_score, groups, exclude_groups)
    scores = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2 or len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not scores:
        return float("nan")
    return float(np.median(scores))


def median_auprc_per_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
) -> float:
    """Median AUPRC over groups; robust to one badly predicted cell line.

    Parameters
    ----------
    y_true, y_score : array-like of shape (n_samples,)
        Binary labels and continuous model scores.
    groups : array-like of shape (n_samples,)
        Group label per sample.
    exclude_groups : set of str, optional keyword-only
        Groups dropped before the median.

    Returns
    -------
    float
        Median per-group average precision, or ``nan`` when no group is
        eligible.

    Notes
    -----
    Same single-class caveat as :func:`macro_auprc_per_group`, but the median
    makes the statistic insensitive to one degenerate group as long as fewer
    than half the groups are degenerate.
    """
    y_true, y_score, groups = _apply_group_exclude(y_true, y_score, groups, exclude_groups)
    scores = []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    if not scores:
        return float("nan")
    return float(np.median(scores))


# Names expected by train.py
Spearman_Group = spearman_group
macro_auroc = macro_auroc_per_group
macro_auprc = macro_auprc_per_group
median_auroc = median_auroc_per_group
median_auprc = median_auprc_per_group


def per_group_auroc_auprc(
    y_score: np.ndarray, y_true: np.ndarray, groups: np.ndarray
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Per-group AUROC and AUPRC, returned as dictionaries keyed by group.

    Parameters
    ----------
    y_score : array-like of shape (n_samples,)
        Continuous model score. **Note the argument order**: unlike the other
        helpers in this module the score comes first and the label second.
    y_true : array-like of shape (n_samples,)
        Binary ground truth.
    groups : array-like of shape (n_samples,)
        Group label per sample (typically the cell line).

    Returns
    -------
    auroc_by : dict
        Group -> AUROC. Groups with a single class map to ``nan``; the key is
        still present, so the dictionary mirrors the group set.
    auprc_by : dict
        Group -> AUPRC.

    Notes
    -----
    Groups with fewer than 2 samples are dropped from both dictionaries, so the
    two dictionaries always share the same key set. This function is the
    detailed companion of the scalar macro/median helpers: use it when the
    per-cell-line breakdown has to be reported, not just the summary.
    """
    auroc_by: dict[Any, float] = {}
    auprc_by: dict[Any, float] = {}
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) < 2:
            continue
        yt = np.asarray(y_true[idx]).astype(int)
        ys = np.asarray(y_score[idx]).astype(float)
        if len(np.unique(yt)) >= 2:
            auroc_by[g] = float(roc_auc_score(yt, ys))
        else:
            auroc_by[g] = float("nan")
        auprc_by[g] = float(average_precision_score(yt, ys))
    return auroc_by, auprc_by


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of true positives among the ``k`` highest-scoring samples.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground truth, cast to ``int``.
    y_score : array-like of shape (n_samples,)
        Continuous model score; ranked in descending order.
    k : int
        Cut-off. Clipped to ``n_samples``.

    Returns
    -------
    float
        ``positives in top-k / k``, or ``nan`` for an empty input.

    Notes
    -----
    * ``k`` is clipped rather than rejected. When ``k > n_samples`` the metric
      degenerates to the positive rate of the whole set, which can look
      deceptively good on small evaluation sets (the golden-intersection sets
      are only a few hundred loci).
    * Ties are broken by ``np.argsort``'s ordering, which is not guaranteed
      stable, so on heavily tied scores (e.g. binarised outputs) the value is
      implementation-dependent.
    * This is the "top-k precision" metric of Supplementary Methods §7.1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = int(y_true.size)
    if n == 0:
        return float("nan")
    k = min(int(k), n)
    order = np.argsort(-y_score)
    top = y_true[order[:k]]
    return float(np.sum(top) / k)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Fraction of all positives recovered within the ``k`` highest scores.

    Parameters
    ----------
    y_true, y_score : array-like of shape (n_samples,)
        Binary ground truth and continuous model score.
    k : int
        Cut-off, clipped to ``n_samples``.

    Returns
    -------
    float
        ``positives in top-k / total positives``, or ``nan`` when the set
        contains no positive sample (the denominator would be zero).

    Notes
    -----
    Recall@k is not among the metrics enumerated in Supplementary Methods §7.1
    (AUROC, AUPRC, precision@K); it is reported as an auxiliary view of the same
    ranking, and it is the complement of precision@k only when ``k`` equals the
    number of positives.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return float("nan")
    k = min(int(k), int(y_true.size))
    order = np.argsort(-y_score)
    top = y_true[order[:k]]
    return float(np.sum(top) / n_pos)


# Deliberately minimal variant. It is kept separate from
# `ranking_metrics_bundle` so that logging call sites which only need the two
# scalars do not pay for the per-k sorting performed by precision_at_k /
# recall_at_k.
def ranking_metrics_minimal(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUROC and AUPRC only (no top-k or label rate).

    Parameters
    ----------
    y_true, y_score : array-like of shape (n_samples,)
        Binary ground truth and continuous model score.

    Returns
    -------
    dict of str -> float
        Two keys, ``"auroc"`` and ``"auprc"``; either may be ``nan``.

    Notes
    -----
    Kept separate from :func:`ranking_metrics_bundle` so that call sites which
    log only the two scalars avoid the per-``k`` sorting done by
    :func:`precision_at_k` and :func:`recall_at_k`.
    """
    return {
        "auroc": auroc_safe(y_true, y_score),
        "auprc": auprc_safe(y_true, y_score),
    }


def ranking_metrics_bundle(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    k_list: Sequence[int] | None = None,
) -> dict[str, float]:
    """AUROC, AUPRC, positive rate and precision/recall@k in one dictionary.

    Parameters
    ----------
    y_true, y_score : array-like of shape (n_samples,)
        Binary ground truth and continuous model score.
    k_list : sequence of int, optional keyword-only
        Cut-offs for the top-k metrics; defaults to ``[100, 500, 1000]``, the
        scale of the genome-wide benchmark sets. An empty sequence returns only
        the three scalar metrics.

    Returns
    -------
    dict of str -> float
        ``auroc``, ``auprc``, ``label_positive_rate`` plus ``precision_at_{k}``
        and ``recall_at_{k}`` for every ``k``.

    Notes
    -----
    Key names are built by f-string interpolation, so downstream lookups must use
    a plain integer ``k`` (a float would stringify differently).
    ``label_positive_rate`` is included so that AUPRC values from datasets with
    different class balance can be compared: AUPRC's no-skill baseline is
    exactly that rate.
    """
    k_list = list(k_list or [100, 500, 1000])
    out: dict[str, float] = {
        "auroc": auroc_safe(y_true, y_score),
        "auprc": auprc_safe(y_true, y_score),
        "label_positive_rate": float(np.mean(np.asarray(y_true).astype(float))),
    }
    if not k_list:
        return out
    for k in k_list:
        out[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
        out[f"recall_at_{k}"] = recall_at_k(y_true, y_score, k)
    return out


def make_val_score_fn(
    metric: str,
    combo_weights: dict[str, float] | None = None,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
):
    """Build the validation-score callable used for early stopping / selection.

    Every callable returned here has the signature
    ``f(model, val_data: dict, device: str) -> float`` and follows the "higher is
    better" convention. It is invoked by the training loop with the validation
    batch dictionary and the device string, performs one ``model.predict`` pass
    (``verbose=False``) and reduces the predictions to a single scalar.

    Parameters
    ----------
    metric : str
        Metric selector, case- and whitespace-insensitive. Accepted:
        ``"spearman"`` / ``"spearman_group"``, ``"auroc"`` / ``"macro_auroc"`` /
        ``"auroc_macro"``, ``"auprc"`` / ``"macro_auprc"`` / ``"auprc_macro"``,
        ``"median_auroc"`` / ``"auroc_median"``, ``"median_auprc"`` /
        ``"auprc_median"``, ``"combo"`` / ``"combined"``, and ``"m2a_combo"`` /
        ``"median_auroc_auprc_combo"``.
    combo_weights : dict of str -> float, optional
        Weights for the composite selectors (``"combo"``, ``"m2a_combo"``); keys
        are metric names as above. ``None`` selects the built-in defaults listed
        in the Notes.
    exclude_groups : set of str, optional keyword-only
        Cell lines excluded from the macro/median variants.

    Returns
    -------
    callable
        ``f(model, val_data, device) -> float``.

    Raises
    ------
    ValueError
        When ``metric`` is not one of the accepted selectors.

    Notes
    -----
    * ``"combo"`` defaults to ``0.6 * macro_auprc + 0.4 * spearman_group``;
      ``"m2a_combo"`` defaults to ``0.7 * median_auroc + 0.3 * median_auprc``.
      The "m2a" composite is the checkpoint-selection score referred to in
      Supplementary Methods §5.4/§5.5, but the numeric weights are code
      defaults (the formulas appear as figures in the manuscript text).
    * Each composite re-runs ``model.predict`` once per component. Stateless and
      simple, at the cost of redundant inference; this only matters at
      validation frequency.
    * Grouping-key inconsistency: the Spearman selector groups by
      ``val_data["cell_type"]`` while the ranking selectors group by
      ``val_data["cell_line"]``. Both keys must therefore be present.
    * ``_combo`` renormalises by the sum of the weights it actually found, so a
      partial weight dictionary yields a proper weighted average instead of a
      silently down-scaled score.
    """

    def _spearman(model, val_data: dict, device: str) -> float:
        """Mean per-cell-type Spearman between prediction and target.

        Requests ``with_true=True`` so that both the predictions and the
        regression targets come back from ``model.predict``; the grouping key is
        ``val_data["cell_type"]``.
        """
        preds, y_reg = model.predict(val_data, batch_size=256, device=device, verbose=False, with_true=True)
        return float(spearman_group(preds, y_reg, val_data["cell_type"]))

    def _macro_auroc(model, val_data: dict, device: str) -> float:
        """Macro AUROC over cell lines, with ``exclude_groups`` applied."""
        preds, _y = model.predict(val_data, batch_size=256, device=device, verbose=False, with_true=False)
        return float(
            macro_auroc(
                val_data["label"],
                preds,
                val_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    def _macro_auprc(model, val_data: dict, device: str) -> float:
        """Macro AUPRC over cell lines, with ``exclude_groups`` applied."""
        preds, _y = model.predict(val_data, batch_size=256, device=device, verbose=False, with_true=False)
        return float(
            macro_auprc(
                val_data["label"],
                preds,
                val_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    def _median_auroc(model, val_data: dict, device: str) -> float:
        """Median AUROC over cell lines, with ``exclude_groups`` applied."""
        preds, _y = model.predict(val_data, batch_size=256, device=device, verbose=False, with_true=False)
        return float(
            median_auroc(
                val_data["label"],
                preds,
                val_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    M2A_COMBO_DEFAULT = {"median_auroc": 0.7, "median_auprc": 0.3}
    # Default weights of the "m2a" composite (median AUROC + median AUPRC),
    # i.e. the checkpoint-selection score of Supplementary Methods §5.4/§5.5.
    # Declared before `_combo` only for locality; the `m2a_combo` branch below is
    # the one that consumes it.

    def _combo(model, val_data: dict, device: str) -> float:
        """Weighted sum of the requested sub-metrics, renormalised by their weights.

        ``w`` comes from the enclosing ``combo_weights`` (default
        ``{"macro_auprc": 0.6, "spearman_group": 0.4}``). Only the keys present
        in ``w`` are evaluated, so the value is always a proper weighted average
        over the available components.

        Notes
        -----
        ``_median_auprc`` is referenced here although it is *defined further
        down* in the enclosing function body. That works because Python resolves
        closure variables at call time, not at definition time, but the callable
        therefore cannot be invoked before the enclosing factory has finished
        executing. Unlike the N1/N2/N3 composites, this one has no ``nan`` guard:
        an undefined component propagates to the whole score.
        """
        w = combo_weights or {"macro_auprc": 0.6, "spearman_group": 0.4}
        score = 0.0
        total = 0.0
        if "macro_auprc" in w:
            score += float(w["macro_auprc"]) * _macro_auprc(model, val_data, device)
            total += float(w["macro_auprc"])
        if "macro_auroc" in w:
            score += float(w["macro_auroc"]) * _macro_auroc(model, val_data, device)
            total += float(w["macro_auroc"])
        if "spearman_group" in w:
            score += float(w["spearman_group"]) * _spearman(model, val_data, device)
            total += float(w["spearman_group"])
        if "median_auroc" in w:
            score += float(w["median_auroc"]) * _median_auroc(model, val_data, device)
            total += float(w["median_auroc"])
        if "median_auprc" in w:
            score += float(w["median_auprc"]) * _median_auprc(model, val_data, device)
            total += float(w["median_auprc"])
        return score / max(total, 1e-12)

    def _median_auprc(model, val_data: dict, device: str) -> float:
        """Median AUPRC over cell lines.

        Defined after ``_combo`` (which already refers to it); see the note in
        ``_combo`` on late binding.
        """
        preds, _y = model.predict(val_data, batch_size=256, device=device, verbose=False, with_true=False)
        return float(
            median_auprc(
                val_data["label"],
                preds,
                val_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    # Normalise the selector once so callers can pass e.g. "Macro_AUROC " with
    # arbitrary case and surrounding whitespace; the extra spellings are
    # historical aliases mapping onto the same callable.
    m = str(metric).lower().strip()
    if m in ("spearman", "spearman_group"):
        return _spearman
    if m in ("auroc", "macro_auroc", "auroc_macro"):
        return _macro_auroc
    if m in ("auprc", "macro_auprc", "auprc_macro"):
        return _macro_auprc
    if m in ("median_auroc", "auroc_median"):
        return _median_auroc
    if m in ("median_auprc", "auprc_median"):
        return _median_auprc
    if m in ("combo", "combined"):
        return _combo
    if m in ("m2a_combo", "median_auroc_auprc_combo"):
        def _m2a_combo(model, val_data: dict, device: str) -> float:
            """Weighted median-AUROC + median-AUPRC composite (the "m2a" score).

            ``w`` comes from the enclosing ``combo_weights`` or from
            ``M2A_COMBO_DEFAULT`` (0.7 / 0.3). Only those two keys are consulted, so
            any other weight in a supplied dictionary is ignored here; the result is
            renormalised by the weights actually used.

            Like ``_combo``, this composite has no ``nan`` guard.
            """
            w = combo_weights or M2A_COMBO_DEFAULT
            score = 0.0
            total = 0.0
            if "median_auroc" in w:
                score += float(w["median_auroc"]) * _median_auroc(model, val_data, device)
                total += float(w["median_auroc"])
            if "median_auprc" in w:
                score += float(w["median_auprc"]) * _median_auprc(model, val_data, device)
                total += float(w["median_auprc"])
            return score / max(total, 1e-12)

        return _m2a_combo
    raise ValueError(f"Unknown validation metric: {metric!r}")


def make_n1_val_score_fn(
    valid_data: dict,
    external_batches: dict[str, dict],
    combo_weights: dict[str, float] | None = None,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
    predict_batch_size: int = 256,
):
    """Phase N1: maximize 721 valid median AUROC + K562 external CRISPRi/dualKO AUROC.

    Parameters
    ----------
    valid_data : dict
        Validation batch (the "721" split, i.e. the 7:2:1 locus-aware split of
        Supplementary Methods §2.3.3). Bound here rather than passed per call, so
        the returned callable ignores its ``_val_data`` argument.
    external_batches : dict of str -> dict
        Batches keyed by dataset name; the two names consumed are
        ``"K562_CRISPRi"`` and ``"K562_dualKO"``. A missing key removes that term.
    combo_weights : dict of str -> float, optional
        Weights of the three terms; defaults to
        ``{"median_auroc": 0.34, "k562_crispr_auroc": 0.33,
        "k562_dual_auroc": 0.33}``.
    exclude_groups : set of str, optional keyword-only
        Cell lines excluded from the internal median AUROC.
    predict_batch_size : int, default 256
        Batch size forwarded to ``model.predict``.

    Returns
    -------
    callable
        ``f(model, val_data, device) -> float``, higher is better.

    Notes
    -----
    * The phase label "N1" and the 0.34/0.33/0.33 weights are internal
      training-schedule conventions; the manuscript describes no phased training
      and none of these numbers appear in it.
    * Each term is ``nan``-guarded with ``v == v``, so a term that cannot be
      evaluated (e.g. a single-class external set) is dropped from numerator and
      denominator and the score remains a weighted average of what reported.
    """

    def _median_valid(model, _val_data: dict, device: str) -> float:
        """Median AUROC over cell lines on the internal validation split.

        Reads the bound ``valid_data`` rather than the ``_val_data`` argument,
        which exists only to match the shared callable signature.
        """
        preds, _y = model.predict(
            valid_data, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        return float(
            median_auroc(
                valid_data["label"],
                preds,
                valid_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    def _external_auroc(model, batch: dict, device: str) -> float:
        """Pooled AUROC over all samples of one external batch."""
        preds, _ = model.predict(
            batch, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        return float(auroc_safe(batch["label"], preds))

    w = combo_weights or {
        "median_auroc": 0.34,
        "k562_crispr_auroc": 0.33,
        "k562_dual_auroc": 0.33,
    }

    def _n1(model, _val_data: dict, device: str) -> float:
        """Weighted average of the internal median AUROC and the two external AUROCs.

        Terms that evaluate to ``nan`` (missing batch, single-class target) are
        dropped from both the numerator and the denominator, so the result stays
        a weighted average of the terms that did report. If no term reports, the
        denominator collapses to ``1e-12`` and the score becomes ``0.0``.
        """
        score = 0.0
        total = 0.0
        if "median_auroc" in w:
            v = _median_valid(model, _val_data, device)
            if v == v:
                score += float(w["median_auroc"]) * v
                total += float(w["median_auroc"])
        crispr = external_batches.get("K562_CRISPRi")
        if crispr is not None and "k562_crispr_auroc" in w:
            v = _external_auroc(model, crispr, device)
            if v == v:
                score += float(w["k562_crispr_auroc"]) * v
                total += float(w["k562_crispr_auroc"])
        dual = external_batches.get("K562_dualKO")
        if dual is not None and "k562_dual_auroc" in w:
            v = _external_auroc(model, dual, device)
            if v == v:
                score += float(w["k562_dual_auroc"]) * v
                total += float(w["k562_dual_auroc"])
        return score / max(total, 1e-12)

    return _n1


def compute_n1_success_gates(
    *,
    test_auroc_median: float,
    external_per_dataset: dict[str, dict[str, Any]],
    target: float = 0.70,
) -> dict[str, Any]:
    """Boolean gates for Phase N1 KPIs (721 test median + K562 external AUROCs).

    Parameters
    ----------
    test_auroc_median : float
        Median AUROC on the held-out 721 test split.
    external_per_dataset : dict of str -> dict
        Per-dataset metric dictionaries, e.g.
        ``{"K562_CRISPRi": {"auroc": 0.71}, "K562_dualKO": {"auroc": 0.69}}``.
    target : float, default 0.70
        Single threshold applied to all three gates.

    Returns
    -------
    dict
        The resolved target, the three raw metric values, one boolean per gate
        and the aggregate ``all_n1_gates_pass``. A ``nan`` metric or a missing
        dataset fails its gate, since ``nan`` never compares greater-or-equal.

    Notes
    -----
    The 0.70 target is a project KPI, not a value taken from the manuscript.
    Raw values are reported next to the booleans so that a failing run still
    leaves a diagnosable record.
    """
    k_crispr = float(external_per_dataset.get("K562_CRISPRi", {}).get("auroc", float("nan")))
    k_dual = float(external_per_dataset.get("K562_dualKO", {}).get("auroc", float("nan")))
    med = float(test_auroc_median)
    t = float(target)

    def _ok(x: float) -> bool:
        """``True`` only when ``x`` is a real number reaching ``t``.

        The ``x == x`` test rejects ``nan`` (every comparison with ``nan`` is
        ``False``), so an undefined metric fails its gate rather than silently
        passing.
        """
        return x == x and x >= t

    gates = {
        "target": t,
        "test_median_auroc_ge_target": _ok(med),
        "k562_crispr_auroc_ge_target": _ok(k_crispr),
        "k562_dual_auroc_ge_target": _ok(k_dual),
        "all_n1_gates_pass": _ok(med) and _ok(k_crispr) and _ok(k_dual),
        "test_auroc_median": med,
        "k562_crispr_auroc": k_crispr,
        "k562_dual_auroc": k_dual,
    }
    return gates


def make_n2_val_score_fn(
    valid_data: dict,
    external_batches: dict[str, dict],
    combo_weights: dict[str, float] | None = None,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
    predict_batch_size: int = 256,
    primary_k: int = 100,
):
    """Phase N2: 721 valid median AUPRC + K562 external AUPRC / precision@k (no external train).

    Parameters
    ----------
    valid_data : dict
        Validation batch, bound at construction time (the returned callable
        ignores its ``_val_data`` argument).
    external_batches : dict of str -> dict
        External batches under ``"K562_CRISPRi"`` / ``"K562_dualKO"``.
    combo_weights : dict of str -> float, optional
        Defaults to ``{"median_auprc": 0.40, "k562_crispr_auprc": 0.25,
        "k562_dual_auprc": 0.25, "k562_crispr_precision_at_k": 0.05,
        "k562_dual_precision_at_k": 0.05}``; the two precision keys are expanded
        with ``primary_k``. Unrecognised keys are ignored.
    exclude_groups : set of str, optional keyword-only
        Cell lines excluded from the internal median AUPRC.
    predict_batch_size : int, default 256
        Batch size for ``model.predict``.
    primary_k : int, default 100
        ``k`` of the external precision@k terms.

    Returns
    -------
    callable
        ``f(model, val_data, device) -> float``, higher is better.

    Notes
    -----
    Weights and the "N2" label are internal conventions, not manuscript values.
    The AUPRC terms carry 90% of the mass because AUPRC is the informative
    metric on the strongly imbalanced external K562 sets; the two precision@k
    terms (5% each) act mainly as tie-breakers.
    """

    def _median_valid_auprc(model, _val_data: dict, device: str) -> float:
        """Median AUPRC over cell lines on the bound validation split."""
        preds, _y = model.predict(
            valid_data, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        return float(
            median_auprc(
                valid_data["label"],
                preds,
                valid_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    def _external_rank(model, batch: dict, device: str) -> dict[str, float]:
        """Ranking metrics of one external batch at ``k = primary_k``.

        Returns the dictionary from :func:`ranking_metrics_bundle` restricted to
        the single ``k`` captured from the enclosing scope, so the keys already
        include ``precision_at_{k}``.
        """
        preds, _ = model.predict(
            batch, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        y = np.asarray(batch["label"], dtype=int)
        p = np.asarray(preds, dtype=float)
        return ranking_metrics_bundle(y, p, k_list=[primary_k])

    pk = f"precision_at_{int(primary_k)}"
    w = combo_weights or {
        "median_auprc": 0.40,
        "k562_crispr_auprc": 0.25,
        "k562_dual_auprc": 0.25,
        "k562_crispr_precision_at_k": 0.05,
        "k562_dual_precision_at_k": 0.05,
    }

    def _n2(model, _val_data: dict, device: str) -> float:
        """Weighted average of the internal median AUPRC and the external terms.

        Terms are ``k562_crispr_auprc`` / ``k562_dual_auprc`` (pooled AUPRC) and
        ``k562_crispr_precision_at_k`` / ``k562_dual_precision_at_k`` (top-k
        precision at the captured ``primary_k``). ``nan`` terms are dropped from
        numerator and denominator.
        """
        score = 0.0
        total = 0.0
        if "median_auprc" in w:
            v = _median_valid_auprc(model, _val_data, device)
            if v == v:
                score += float(w["median_auprc"]) * v
                total += float(w["median_auprc"])
        crispr = external_batches.get("K562_CRISPRi")
        if crispr is not None:
            r = _external_rank(model, crispr, device)
            if "k562_crispr_auprc" in w:
                v = float(r["auprc"])
                if v == v:
                    score += float(w["k562_crispr_auprc"]) * v
                    total += float(w["k562_crispr_auprc"])
            wk = "k562_crispr_precision_at_k"
            if wk in w and pk in r:
                v = float(r[pk])
                if v == v:
                    score += float(w[wk]) * v
                    total += float(w[wk])
        dual = external_batches.get("K562_dualKO")
        if dual is not None:
            r = _external_rank(model, dual, device)
            if "k562_dual_auprc" in w:
                v = float(r["auprc"])
                if v == v:
                    score += float(w["k562_dual_auprc"]) * v
                    total += float(w["k562_dual_auprc"])
            wk = "k562_dual_precision_at_k"
            if wk in w and pk in r:
                v = float(r[pk])
                if v == v:
                    score += float(w[wk]) * v
                    total += float(w[wk])
        return score / max(total, 1e-12)

    return _n2


def compute_n2_success_gates(
    *,
    test_auprc_median: float,
    test_auroc_median: float | None = None,
    external_per_dataset: dict[str, dict[str, Any]],
    target_auprc: float = 0.15,
    target_auroc: float | None = 0.70,
    primary_k: int = 100,
    target_precision_at_k: float | None = 0.10,
) -> dict[str, Any]:
    """Phase N2 gates: 721 test median AUROC (>=0.70) + median AUPRC + K562 external AUPRC/p@k.

    Parameters
    ----------
    test_auprc_median : float
        Median AUPRC on the test split.
    test_auroc_median : float or None, optional
        Median AUROC on the test split; ``None`` makes the AUROC gate fail.
    external_per_dataset : dict of str -> dict
        Per-dataset metrics for ``"K562_CRISPRi"`` / ``"K562_dualKO"``.
    target_auprc : float, default 0.15
        Threshold for the three AUPRC gates.
    target_auroc : float or None, default 0.70
        Threshold for the test AUROC gate; ``None`` disables that gate.
    primary_k : int, default 100
        ``k`` used to look up ``precision_at_{k}``.
    target_precision_at_k : float or None, default 0.10
        Threshold for the two precision@k gates; ``None`` disables them.

    Returns
    -------
    dict
        Resolved targets, raw metric values (exported as ``None`` where
        undefined so the record stays JSON-friendly), the individual gates and
        ``all_n2_gates_pass``.

    Notes
    -----
    Every threshold here is an engineering target, not a manuscript value. The
    ``None`` handling makes individual gates optional without changing the
    signature; a disabled gate always passes, so ``all_n2_gates_pass`` keeps its
    meaning.
    """
    pk = f"precision_at_{int(primary_k)}"
    crispr = external_per_dataset.get("K562_CRISPRi", {})
    dual = external_per_dataset.get("K562_dualKO", {})
    med = float(test_auprc_median)
    c_auprc = float(crispr.get("auprc", float("nan")))
    d_auprc = float(dual.get("auprc", float("nan")))
    c_pk = float(crispr.get(pk, float("nan")))
    d_pk = float(dual.get(pk, float("nan")))
    ta = float(target_auprc)

    def _ok_auprc(x: float) -> bool:
        """``nan``-safe ``x >= ta`` test for the AUPRC gates."""
        return x == x and x >= ta

    def _ok_pk(x: float) -> bool:
        """Precision@k gate; ``target_precision_at_k=None`` disables it."""
        if target_precision_at_k is None:
            return True
        return x == x and x >= float(target_precision_at_k)

    def _ok_auroc(x: float | None) -> bool:
        """AUROC gate; ``target_auroc=None`` disables it, a missing value fails it."""
        if target_auroc is None:
            return True
        if x is None or x != x:
            return False
        return float(x) >= float(target_auroc)

    med_auroc = float(test_auroc_median) if test_auroc_median is not None else float("nan")

    gates = {
        "target_auprc": ta,
        "target_auroc": target_auroc,
        "primary_k": int(primary_k),
        "target_precision_at_k": target_precision_at_k,
        "test_median_auroc_ge_target": _ok_auroc(med_auroc if med_auroc == med_auroc else None),
        "test_median_auprc_ge_target": _ok_auprc(med),
        "k562_crispr_auprc_ge_target": _ok_auprc(c_auprc),
        "k562_dual_auprc_ge_target": _ok_auprc(d_auprc),
        "k562_crispr_precision_at_k_ge_target": _ok_pk(c_pk),
        "k562_dual_precision_at_k_ge_target": _ok_pk(d_pk),
        "all_n2_gates_pass": (
            _ok_auroc(med_auroc if med_auroc == med_auroc else None)
            and _ok_auprc(med)
            and _ok_auprc(c_auprc)
            and _ok_auprc(d_auprc)
            and _ok_pk(c_pk)
            and _ok_pk(d_pk)
        ),
        "test_auroc_median": med_auroc if med_auroc == med_auroc else None,
        "test_auprc_median": med,
        "k562_crispr_auprc": c_auprc,
        "k562_dual_auprc": d_auprc,
        pk + "_crispr": c_pk,
        pk + "_dual": d_pk,
    }
    return gates


def make_ext_pk_val_score_fn(
    valid_data: dict,
    external_batches: dict[str, dict],
    combo_weights: dict[str, float] | None = None,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
    predict_batch_size: int = 256,
    primary_k: int = 10,
    secondary_k: int | None = 100,
    earlystop_k_list: list[int] | None = None,
):
    """Phase N3/N4: 721 valid median + K562 external macro precision@k.

    Parameters
    ----------
    valid_data : dict
        Validation batch, bound at construction time.
    external_batches : dict of str -> dict
        External batches under ``"K562_CRISPRi"`` / ``"K562_dualKO"``.
    combo_weights : dict of str -> float, optional
        Metric-name to weight mapping. Supported names: ``"median_auprc"``,
        ``"median_auroc"``, ``"external_macro_precision_at_{k}"`` (mean over the
        external datasets), ``"k562_crispr_precision_at_{k}"``,
        ``"k562_dual_precision_at_{k}"``, ``"k562_crispr_auroc"`` and
        ``"k562_dual_auroc"``. Defaults to 0.08 median AUPRC plus 0.23 for each
        of the four p@10 / p@100 terms. Unrecognised keys are skipped silently.
    exclude_groups : set of str, optional keyword-only
        Cell lines excluded from the internal median metrics.
    predict_batch_size : int, default 256
        Batch size for ``model.predict``.
    primary_k : int, default 10
        Small cut-off for the precision@k terms.
    secondary_k : int or None, default 100
        Large cut-off; ``None`` falls back to 100, it does *not* disable the
        term.
    earlystop_k_list : list of int, optional
        Accepted for interface symmetry with the other factories but **not used
        here**; callers pass it and it is silently ignored.

    Returns
    -------
    callable
        ``f(model, val_data, device) -> float``, higher is better.

    Notes
    -----
    ``_ext_pk`` parses the cut-off out of the metric key with
    ``key.rsplit("_", 1)[-1]``, so the numeric suffix must be the last
    underscore-separated token. Terms whose dataset is absent or whose metric is
    ``nan`` are dropped from numerator and denominator.
    """

    def _median_valid_auprc(model, _val_data: dict, device: str) -> float:
        """Median AUPRC over cell lines on the bound validation split."""
        preds, _y = model.predict(
            valid_data, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        return float(
            median_auprc(
                valid_data["label"],
                preds,
                valid_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    def _median_valid_auroc(model, _val_data: dict, device: str) -> float:
        """Median AUROC over cell lines on the bound validation split."""
        preds, _y = model.predict(
            valid_data, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        return float(
            median_auroc(
                valid_data["label"],
                preds,
                valid_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

    def _external_pk(model, batch: dict, device: str, k: int) -> float:
        """Precision@k of one external batch; ``k`` is clipped to the batch size."""
        preds, _ = model.predict(
            batch, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        y = np.asarray(batch["label"], dtype=int)
        p = np.asarray(preds, dtype=float)
        return float(precision_at_k(y, p, int(k)))

    def _external_auroc(model, batch: dict, device: str) -> float:
        """Pooled AUROC over all samples of one external batch."""
        preds, _ = model.predict(
            batch, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        y = np.asarray(batch["label"], dtype=int)
        p = np.asarray(preds, dtype=float)
        return float(auroc_safe(y, p))

    def _macro_external_pk(model, device: str, k: int) -> float:
        """Mean precision@k over whichever external datasets are available.

        Returns ``nan`` when neither ``"K562_CRISPRi"`` nor ``"K562_dualKO"`` is
        present, which the caller treats as "term unavailable".
        """
        vals = []
        for name in ("K562_CRISPRi", "K562_dualKO"):
            batch = external_batches.get(name)
            if batch is None:
                continue
            v = _external_pk(model, batch, device, k)
            if v == v:
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    pk = int(primary_k)
    sk = int(secondary_k) if secondary_k is not None else 100

    default_w: dict[str, float] = {
        "median_auprc": 0.08,
        f"k562_crispr_precision_at_{pk}": 0.23,
        f"k562_dual_precision_at_{pk}": 0.23,
        f"k562_crispr_precision_at_{sk}": 0.23,
        f"k562_dual_precision_at_{sk}": 0.23,
    }

    w = combo_weights or default_w

    def _ext_pk(model, _val_data: dict, device: str) -> float:
        """Weighted average of the internal median metrics and the external terms.

        The weight dictionary encodes both the metric name and, for precision@k
        terms, the cut-off, e.g. ``"k562_dual_precision_at_30": 0.1``. Supported
        key forms are listed in the factory docstring; unrecognised keys are
        skipped silently. Terms whose metric is ``nan`` are dropped from
        numerator and denominator.
        """
        score = 0.0
        total = 0.0
        if "median_auprc" in w:
            v = _median_valid_auprc(model, _val_data, device)
            if v == v:
                score += float(w["median_auprc"]) * v
                total += float(w["median_auprc"])
        if "median_auroc" in w:
            v = _median_valid_auroc(model, _val_data, device)
            if v == v:
                score += float(w["median_auroc"]) * v
                total += float(w["median_auroc"])
        for key, wt in w.items():
            if key in ("median_auprc", "median_auroc") or wt == 0:
                continue
            v = float("nan")
            if key.startswith("external_macro_precision_at_"):
                try:
                    k = int(key.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                v = _macro_external_pk(model, device, k)
            elif key.startswith("k562_crispr_precision_at_"):
                try:
                    k = int(key.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                batch = external_batches.get("K562_CRISPRi")
                if batch is not None:
                    v = _external_pk(model, batch, device, k)
            elif key.startswith("k562_dual_precision_at_"):
                try:
                    k = int(key.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                batch = external_batches.get("K562_dualKO")
                if batch is not None:
                    v = _external_pk(model, batch, device, k)
            elif key == "k562_crispr_auroc":
                batch = external_batches.get("K562_CRISPRi")
                if batch is not None:
                    v = _external_auroc(model, batch, device)
            elif key == "k562_dual_auroc":
                batch = external_batches.get("K562_dualKO")
                if batch is not None:
                    v = _external_auroc(model, batch, device)
            if v == v:
                score += float(wt) * v
                total += float(wt)
        return score / max(total, 1e-12)

    return _ext_pk


def locus_key_from_id(row_id: str) -> str:
    """Extract a genomic locus key from an arbitrary row identifier.

    Parameters
    ----------
    row_id : str
        Sample identifier, e.g. an ENCODE cCRE accession (``EH38E3392707``) or a
        composite id carrying coordinates.

    Returns
    -------
    str
        The matched locus key, or ``str(row_id)`` unchanged when nothing
        matches.

    Notes
    -----
    **The regular expression used here cannot match the identifiers this project
    actually uses, so in practice the function always returns
    ``str(row_id)``.** Two independent reasons: (i) the pattern is written in a
    raw string with *doubled* backslashes, which makes it expect a literal
    backslash followed by the letter "d" instead of a digit character class;
    and (ii) even the intended pattern separates the chromosome from the
    coordinates with an underscore, whereas the project's coordinate keys use a
    colon and a hyphen (e.g. ``chr2:197414404-197414904``). This is recorded as
    documentation only -- the file is not being corrected here.

    The consequence is benign at the current call site: both external K562
    batches use the same cCRE accession as ``id``, so the identity fallback still
    intersects the two screens on the right key. It would break for a caller that
    relies on collapsing different ids belonging to the same locus.
    """
    m = re.search(r"(chr\\d+_\\d+-\\d+)", str(row_id))
    return m.group(1) if m else str(row_id)


def golden_intersection_labels_preds(
    crispr_batch: dict[str, Any],
    dual_batch: dict[str, Any],
    scores_cr: np.ndarray,
    scores_du: np.ndarray,
    *,
    neg_threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray] | None:
    """CRISPRi ∩ dualKO loci; label=1 when both neg|score < neg_threshold.

    The two external K562 screens (CRISPRi and dual-SpCas9 knockout) are
    intersected on the locus key; only loci present in both are retained.

    Parameters
    ----------
    crispr_batch, dual_batch : dict
        External batches. Required keys: ``"id"`` (row identifier) and
        ``"neg_score_raw"`` (raw experimental score of the
        phenotype-suppressing / negative-selection direction).
    scores_cr, scores_du : numpy.ndarray
        Model predictions for the two batches, aligned row by row with the
        corresponding ``"id"`` array.
    neg_threshold : float, default 0.03
        Threshold on the raw experimental score; a locus is labelled 1 when its
        value is *below* the threshold in both screens. Keyword-only.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray) or None
        ``(y, pred)`` with ``y`` the integer label and ``pred`` the mean of the
        two model scores per shared locus. ``None`` when a required batch key is
        missing or the screens share no locus.

    Notes
    -----
    * "Golden" means two independent assays agree: a locus is a trusted positive
      when both screens report a small raw score. The column name points at the
      *negative-selection* branch of the MAGeCK RRA score (Supplementary Methods
      §1.4 describes one score per phenotype direction); reading the value as a
      p-value is the only reading under which a high AUROC / precision@k from
      :func:`compute_golden_intersection_metrics` means "model agrees with the
      experiments". Neither the 0.03 threshold nor this intersection design is
      described in the manuscript.
    * Duplicate ids silently overwrite: the dictionaries keep the *last* row seen
      for a locus.
    * ``zip`` truncates to the shortest of ``id`` / score / ``neg_score_raw``, so
      misaligned inputs lose samples instead of raising.
    """
    ids_cr = crispr_batch.get("id")
    ids_du = dual_batch.get("id")
    neg_cr = crispr_batch.get("neg_score_raw")
    neg_du = dual_batch.get("neg_score_raw")
    if ids_cr is None or ids_du is None or neg_cr is None or neg_du is None:
        return None

    locus_cr: dict[str, tuple[float, float]] = {}
    for i, s, n in zip(ids_cr, scores_cr, neg_cr):
        loc = locus_key_from_id(str(i))
        locus_cr[loc] = (float(s), float(n))

    locus_du: dict[str, tuple[float, float]] = {}
    for i, s, n in zip(ids_du, scores_du, neg_du):
        loc = locus_key_from_id(str(i))
        locus_du[loc] = (float(s), float(n))

    common = sorted(set(locus_cr) & set(locus_du))
    if not common:
        return None

    ys: list[int] = []
    preds: list[float] = []
    for loc in common:
        s_cr, n_cr = locus_cr[loc]
        s_du, n_du = locus_du[loc]
        ys.append(int(n_cr < neg_threshold and n_du < neg_threshold))
        preds.append((s_cr + s_du) / 2.0)
    return np.asarray(ys, dtype=int), np.asarray(preds, dtype=float)


def compute_golden_intersection_metrics(
    crispr_batch: dict[str, Any],
    dual_batch: dict[str, Any],
    scores_cr: np.ndarray,
    scores_du: np.ndarray,
    *,
    neg_threshold: float = 0.03,
    k_list: Sequence[int] | None = None,
) -> dict[str, float] | None:
    """Metrics on the CRISPRi/dualKO "golden" locus set.

    Parameters
    ----------
    crispr_batch, dual_batch : dict
        External batches; required keys are documented in
        :func:`golden_intersection_labels_preds`.
    scores_cr, scores_du : numpy.ndarray
        Model predictions for the two batches, aligned with their ``"id"``
        arrays.
    neg_threshold : float, default 0.03
        Label threshold on the raw experimental score (keyword-only).
    k_list : sequence of int, optional keyword-only
        Cut-offs for precision@k; defaults to ``(5, 10, 15, 20, 25, 30)``, which
        suits the small size of the intersection.

    Returns
    -------
    dict of str -> float or None
        ``n``, ``n_pos``, ``pos_rate``, ``golden_auroc`` and, for every ``k``,
        both ``golden_precision_at_{k}`` and the duplicated short alias
        ``p@{k}``. ``None`` when the intersection is empty.

    Notes
    -----
    * The short alias is written from the same computation because report code
      reads ``p@{k}`` while gate code reads ``golden_precision_at_{k}``.
    * When ``k`` exceeds the number of shared loci, :func:`precision_at_k` clips
      ``k`` to the set size, so the value degrades to the positive rate; do not
      compare it across intersections of different sizes.
    * ``golden_auroc`` is ``nan`` when all shared loci fall on one side of the
      threshold.
    """
    merged = golden_intersection_labels_preds(
        crispr_batch,
        dual_batch,
        scores_cr,
        scores_du,
        neg_threshold=neg_threshold,
    )
    if merged is None:
        return None
    y, pred = merged
    ks = list(k_list or (5, 10, 15, 20, 25, 30))
    out: dict[str, float] = {
        "n": float(len(y)),
        "n_pos": float(int(np.sum(y))),
        "pos_rate": float(np.mean(y)) if y.size else float("nan"),
        "golden_auroc": auroc_safe(y, pred),
    }
    for k in ks:
        out[f"golden_precision_at_{int(k)}"] = precision_at_k(y, pred, int(k))
        out[f"p@{int(k)}"] = out[f"golden_precision_at_{int(k)}"]
    return out


def make_golden_intersection_val_score_fn(
    valid_data: dict,
    external_batches: dict[str, dict],
    combo_weights: dict[str, float] | None = None,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
    predict_batch_size: int = 256,
    neg_threshold: float = 0.03,
    earlystop_k_list: list[int] | None = None,
):
    """Phase N3 hPSC retrain: intersection golden @neg_threshold p@k early-stop.

    Parameters
    ----------
    valid_data : dict
        Validation batch, bound at construction time.
    external_batches : dict of str -> dict
        Must contain both ``"K562_CRISPRi"`` and ``"K562_dualKO"``; when either
        is missing the callable returns ``nan``.
    combo_weights : dict of str -> float, optional
        Defaults to ``{"median_auprc": 0.15}`` plus ``0.85 / len(k_list)`` for
        each ``golden_precision_at_{k}`` key. ``"golden_auroc"`` is accepted but
        weighted 0.0 by default.
    exclude_groups : set of str, optional keyword-only
        Cell lines excluded from the internal median AUPRC.
    predict_batch_size : int, default 256
        Batch size for ``model.predict``.
    neg_threshold : float, default 0.03
        Label threshold forwarded to the golden-intersection metrics.
    earlystop_k_list : list of int, optional
        Cut-offs for the golden precision@k terms; defaults to
        ``[5, 10, 15, 20, 25, 30]``.

    Returns
    -------
    callable
        ``f(model, val_data, device) -> float``, higher is better.

    Notes
    -----
    The external predictions are computed once per call and reused by every
    golden metric, so the overhead is two forward passes regardless of
    ``len(k_list)``. The 0.15 / 0.85 split makes early stopping follow the golden
    precision@k terms; that weighting is an experiment-design choice, not a
    manuscript value.
    """

    def _median_valid_auprc(model, _val_data: dict, device: str) -> float:
        """Median AUPRC over cell lines on the bound validation split."""
        preds, _y = model.predict(
            valid_data, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        return float(
            median_auprc(
                valid_data["label"],
                preds,
                valid_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )
    # 0.85 of the composite is spread evenly over the golden precision@k terms
    # and 0.15 comes from the internal median AUPRC; `golden_auroc` is accepted
    # but weighted 0.0, i.e. reported for diagnostics only. These weights are an
    # experiment-design choice, not a manuscript value.

    k_list = [int(x) for x in (earlystop_k_list or [5, 10, 15, 20, 25, 30])]
    default_w: dict[str, float] = {"median_auprc": 0.15, "golden_auroc": 0.0}
    for k in k_list:
        default_w[f"golden_precision_at_{k}"] = 0.85 / max(len(k_list), 1)
    w = combo_weights or default_w

    def _golden_intersection(model, _val_data: dict, device: str) -> float:
        """Weighted average of golden precision@k and the internal median AUPRC.

        The two external forward passes are executed once here and shared by
        every golden metric, so the extra cost does not grow with
        ``len(k_list)``. Returns ``nan`` when either external batch is missing.
        """
        crispr = external_batches.get("K562_CRISPRi")
        dual = external_batches.get("K562_dualKO")
        if crispr is None or dual is None:
            return float("nan")

        pred_cr, _ = model.predict(
            crispr, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        pred_du, _ = model.predict(
            dual, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        met = compute_golden_intersection_metrics(
            crispr,
            dual,
            np.asarray(pred_cr, dtype=float),
            np.asarray(pred_du, dtype=float),
            neg_threshold=neg_threshold,
            k_list=k_list,
        )
        if met is None:
            return float("nan")

        score = 0.0
        total = 0.0
        if "median_auprc" in w:
            v = _median_valid_auprc(model, _val_data, device)
            if v == v:
                score += float(w["median_auprc"]) * v
                total += float(w["median_auprc"])
        for key, wt in w.items():
            if key in ("median_auprc",) or wt == 0:
                continue
            v = float("nan")
            if key == "golden_auroc":
                v = float(met.get("golden_auroc", float("nan")))
            elif key.startswith("golden_precision_at_"):
                v = float(met.get(key, float("nan")))
            if v == v:
                score += float(wt) * v
                total += float(wt)
        return score / max(total, 1e-12)

    return _golden_intersection


def compute_n3_ext_pk10_success_gates(
    *,
    test_auroc_median: float | None,
    external_per_dataset: dict[str, dict[str, Any]],
    target_precision_at_k: float = 0.5,
    target_precision_at_k_crispr: float | None = None,
    target_precision_at_k_dual: float | None = None,
    primary_k: int = 10,
    secondary_k: int | None = 100,
    target_precision_at_k_secondary: float = 0.3,
    baseline_test_median_auroc: float | None = 0.706,
) -> dict[str, Any]:
    """Phase N3: each external dataset p@10 and p@secondary_k must meet targets.

    Parameters
    ----------
    test_auroc_median : float or None
        Median AUROC on the test split; used only to detect regression against
        ``baseline_test_median_auroc``.
    external_per_dataset : dict of str -> dict
        Per-dataset metrics for ``"K562_CRISPRi"`` / ``"K562_dualKO"``.
    target_precision_at_k : float, default 0.5
        Shared threshold for the small-cut-off gate.
    target_precision_at_k_crispr, target_precision_at_k_dual : float or None
        Per-dataset overrides; resolution order is CRISPRi, then dualKO, then the
        shared value.
    primary_k, secondary_k : int
        The two cut-offs (defaults 10 and 100).
    target_precision_at_k_secondary : float, default 0.3
        Threshold for the large-cut-off gate.
    baseline_test_median_auroc : float or None, default 0.706
        Reference median AUROC; a lower test value sets
        ``test_median_auroc_below_baseline``. ``None`` disables the check.

    Returns
    -------
    dict
        Raw values for both datasets, the resolved targets and cut-offs, four
        boolean gates, their conjunction and the regression flag.
    """
    pk_col = f"precision_at_{int(primary_k)}"
    sk = int(secondary_k) if secondary_k is not None else 100
    pk_sec = f"precision_at_{sk}"
    crispr = external_per_dataset.get("K562_CRISPRi", {})
    dual = external_per_dataset.get("K562_dualKO", {})
    c_pk = float(crispr.get(pk_col, float("nan")))
    d_pk = float(dual.get(pk_col, float("nan")))
    c_psec = float(crispr.get(pk_sec, float("nan")))
    d_psec = float(dual.get(pk_sec, float("nan")))
    tgt10 = float(
        target_precision_at_k_crispr
        if target_precision_at_k_crispr is not None
        else (target_precision_at_k_dual if target_precision_at_k_dual is not None else target_precision_at_k)
    )
    tgt100 = float(target_precision_at_k_secondary)

    def _macro_at(col: str) -> float:
        """Mean over the two external datasets, ignoring missing values."""
        c_v = float(crispr.get(col, float("nan")))
        d_v = float(dual.get(col, float("nan")))
        vals = [x for x in (c_v, d_v) if x == x]
        return float(np.mean(vals)) if vals else float("nan")

    def _ok_pk(x: float, tgt: float) -> bool:
        """``nan``-safe ``x >= tgt`` test with the threshold passed explicitly."""
        return x == x and x >= float(tgt)

    ok_c10 = _ok_pk(c_pk, tgt10)
    ok_d10 = _ok_pk(d_pk, tgt10)
    ok_c100 = _ok_pk(c_psec, tgt100)
    ok_d100 = _ok_pk(d_psec, tgt100)

    med_auroc = float(test_auroc_median) if test_auroc_median is not None else float("nan")
    baseline = baseline_test_median_auroc
    regression = None
    if baseline is not None and med_auroc == med_auroc:
        regression = float(med_auroc) < float(baseline)

    gates: dict[str, Any] = {
        "primary_k": int(primary_k),
        "secondary_k": sk,
        "target_precision_at_k": tgt10,
        "target_precision_at_k_secondary": tgt100,
        "external_macro_precision_at_k": _macro_at(pk_col),
        "external_macro_precision_at_secondary_k": _macro_at(pk_sec),
        "k562_crispr_precision_at_k": c_pk,
        "k562_dual_precision_at_k": d_pk,
        "k562_crispr_precision_at_secondary_k": c_psec,
        "k562_dual_precision_at_secondary_k": d_psec,
        "k562_crispr_precision_at_k_ge_target": ok_c10,
        "k562_dual_precision_at_k_ge_target": ok_d10,
        "k562_crispr_precision_at_secondary_k_ge_target": ok_c100,
        "k562_dual_precision_at_secondary_k_ge_target": ok_d100,
        "all_n3_ext_pk_gates_pass": bool(ok_c10 and ok_d10 and ok_c100 and ok_d100),
        "test_auroc_median": med_auroc if med_auroc == med_auroc else None,
        "baseline_test_median_auroc": baseline,
        "test_median_auroc_below_baseline": regression,
    }
    return gates


# Project KPI thresholds for phase N6 (per-dataset precision@k).
# These are tuning targets chosen for the internal training campaign; the
# manuscript (Supplementary Methods §7.1) names the metrics but defines no such
# threshold table.
N6_DEFAULT_PK_THRESHOLDS: dict[int, float] = {10: 0.6, 20: 0.5, 30: 0.45, 40: 0.4}

N6_DEFAULT_COMBO_WEIGHTS: dict[str, float] = {
    "k562_crispr_auroc": 0.15,
    "k562_dual_auroc": 0.15,
    "k562_crispr_precision_at_10": 0.0875,
    "k562_dual_precision_at_10": 0.0875,
    "k562_crispr_precision_at_20": 0.0875,
    "k562_dual_precision_at_20": 0.0875,
    "k562_crispr_precision_at_30": 0.0875,
    "k562_dual_precision_at_30": 0.0875,
    "k562_crispr_precision_at_40": 0.0875,
    "k562_dual_precision_at_40": 0.0875,
}
# Default phase-N6 composite weights. The eight 0.0875 terms plus the two 0.15
# AUROC terms sum to exactly 1.0, so the score is a convex combination and stays
# on the [0, 1] scale of the underlying metrics. The k values here
# (10/20/30/40) intentionally differ from the primary_k/secondary_k arguments
# that `make_n6_val_score_fn` forwards to `make_ext_pk_val_score_fn`.


def compute_n6_per_dataset_gates(
    *,
    test_auroc_median: float | None = None,
    external_per_dataset: dict[str, dict[str, Any]],
    target_auroc: float = 0.7,
    target_p_at_k: dict[int, float] | None = None,
    baseline_test_median_auroc: float | None = 0.706,
    strict_pk_gt: bool = True,
) -> dict[str, Any]:
    """Per-dataset external AUROC + p@10/20/30/40 gates (selection-only tuning targets).

    Parameters
    ----------
    test_auroc_median : float or None, optional
        Median AUROC on the test split; used only for the regression flag.
    external_per_dataset : dict of str -> dict
        Per-dataset metrics for ``"K562_CRISPRi"`` / ``"K562_dualKO"``.
    target_auroc : float, default 0.7
        AUROC threshold for each dataset.
    target_p_at_k : dict of int -> float, optional
        Per-cut-off precision thresholds; defaults to
        :data:`N6_DEFAULT_PK_THRESHOLDS`.
    baseline_test_median_auroc : float or None, default 0.706
        Reference value for the regression flag.
    strict_pk_gt : bool, default True
        Compare precision with ``>`` (strict) instead of ``>=``.

    Returns
    -------
    dict
        One sub-dictionary per dataset (``"K562_CRISPRi"``, ``"K562_dualKO"``)
        holding raw values, per-gate booleans and pass counters, plus the
        aggregate helpers ``max_n_gates_pass`` / ``sum_n_gates_pass`` and the
        regression flag.

    Notes
    -----
    ``strict_pk_gt`` defaults to True, so a precision exactly equal to its target
    counts as a *failure*: a threshold is only credited when beaten.
    ``n_gates_total`` is the AUROC gate plus one gate per entry of
    ``target_p_at_k``.
    """
    pk_tgts = dict(target_p_at_k or N6_DEFAULT_PK_THRESHOLDS)

    def _dataset_checks(ds_name: str) -> dict[str, Any]:
        """Evaluate every gate for one external dataset.

        Returns a sub-dictionary holding the raw metric values (``None`` where
        undefined), one boolean per gate and the pass counters used by the
        aggregates in the caller.
        """
        ds = external_per_dataset.get(ds_name, {})
        out: dict[str, Any] = {"dataset": ds_name}
        n_pass = 0
        auroc = float(ds.get("auroc", float("nan")))
        ok_auroc = auroc == auroc and auroc >= float(target_auroc)
        out["auroc"] = auroc if auroc == auroc else None
        out["auroc_ge_target"] = ok_auroc
        if ok_auroc:
            n_pass += 1
        for k, tgt in sorted(pk_tgts.items()):
            col = f"precision_at_{int(k)}"
            v = float(ds.get(col, float("nan")))
            if strict_pk_gt:
                ok = v == v and v > float(tgt)
            else:
                ok = v == v and v >= float(tgt)
            out[col] = v if v == v else None
            out[f"{col}_ge_target"] = ok
            if ok:
                n_pass += 1
        out["n_gates_pass"] = n_pass
        out["n_gates_total"] = 1 + len(pk_tgts)
        out["all_gates_pass"] = n_pass == out["n_gates_total"]
        return out

    crispr = _dataset_checks("K562_CRISPRi")
    dual = _dataset_checks("K562_dualKO")
    med_auroc = float(test_auroc_median) if test_auroc_median is not None else float("nan")
    regression = None
    if baseline_test_median_auroc is not None and med_auroc == med_auroc:
        regression = float(med_auroc) < float(baseline_test_median_auroc)

    return {
        "target_auroc": float(target_auroc),
        "target_p_at_k": pk_tgts,
        "K562_CRISPRi": crispr,
        "K562_dualKO": dual,
        "crispr_all_gates_pass": crispr.get("all_gates_pass", False),
        "dual_all_gates_pass": dual.get("all_gates_pass", False),
        "any_dataset_all_gates_pass": bool(
            crispr.get("all_gates_pass") or dual.get("all_gates_pass")
        ),
        "both_datasets_all_gates_pass": bool(
            crispr.get("all_gates_pass") and dual.get("all_gates_pass")
        ),
        "crispr_n_gates_pass": crispr.get("n_gates_pass"),
        "dual_n_gates_pass": dual.get("n_gates_pass"),
        "max_n_gates_pass": max(
            int(crispr.get("n_gates_pass", 0)),
            int(dual.get("n_gates_pass", 0)),
        ),
        "sum_n_gates_pass": int(crispr.get("n_gates_pass", 0))
        + int(dual.get("n_gates_pass", 0)),
        "test_auroc_median": med_auroc if med_auroc == med_auroc else None,
        "baseline_test_median_auroc": baseline_test_median_auroc,
        "test_median_auroc_below_baseline": regression,
    }


def make_n6_val_score_fn(
    valid_data: dict,
    external_batches: dict[str, dict],
    combo_weights: dict[str, float] | None = None,
    *,
    exclude_groups: set[str] | frozenset[str] | None = None,
    predict_batch_size: int = 256,
    earlystop_k_list: list[int] | None = None,
):
    """Phase N6: external AUROC + per-dataset p@10/20/30/40 (dual weighted >= crispr).

    Parameters
    ----------
    valid_data, external_batches, combo_weights, exclude_groups,
    predict_batch_size, earlystop_k_list
        Forwarded to :func:`make_ext_pk_val_score_fn`, which defines the accepted
        weight keys. ``earlystop_k_list`` is only recorded there and does not
        influence the score.

    Returns
    -------
    callable
        ``f(model, val_data, device) -> float``, higher is better.

    Notes
    -----
    * Weights come from :data:`N6_DEFAULT_COMBO_WEIGHTS`, which contains only
      external K562 terms, so this phase's score is determined entirely by
      agreement with the external screens.
    * ``primary_k=10`` / ``secondary_k=100`` are forwarded even though the N6
      weight dictionary uses its own cut-offs (10/20/30/40); the forwarded values
      only matter when a caller supplies custom weights.
    * The fallback ``earlystop_k_list or [10, 20, 30, 40]`` has no consumer, as
      noted above.
    """
    _ = valid_data, exclude_groups, earlystop_k_list
    # Deliberate no-op: these names are part of the factory signature but are
    # forwarded explicitly in the call below (valid_data, exclude_groups) or are
    # not consumed at all (earlystop_k_list). The tuple assignment keeps linters
    # quiet without changing the signature.
    w = combo_weights or N6_DEFAULT_COMBO_WEIGHTS
    return make_ext_pk_val_score_fn(
        valid_data,
        external_batches,
        combo_weights=w,
        exclude_groups=exclude_groups,
        predict_batch_size=predict_batch_size,
        primary_k=10,
        secondary_k=100,
        earlystop_k_list=earlystop_k_list or [10, 20, 30, 40],
    )


def compute_n4_success_gates(
    *,
    test_auroc_median: float | None,
    external_per_dataset: dict[str, dict[str, Any]],
    target_test_median_auroc: float = 0.73,
    target_macro_precision_at_10: float = 0.6,
    target_macro_precision_at_30: float = 0.5,
    target_macro_precision_at_50: float = 0.4,
) -> dict[str, Any]:
    """Phase N4: test median AUROC + external macro p@10/30/50 gates.

    Parameters
    ----------
    test_auroc_median : float or None
        Median AUROC on the test split; ``None`` fails the gate.
    external_per_dataset : dict of str -> dict
        Per-dataset metrics for ``"K562_CRISPRi"`` / ``"K562_dualKO"``.
    target_test_median_auroc : float, default 0.73
        AUROC threshold.
    target_macro_precision_at_10, target_macro_precision_at_30,
    target_macro_precision_at_50 : float
        Thresholds for the macro (mean over the two external datasets)
        precision at k = 10, 30 and 50; defaults 0.6, 0.5 and 0.4.

    Returns
    -------
    dict
        Resolved targets, the three macro precision values, the median AUROC,
        the individual gates and ``all_n4_gates_pass``.

    Notes
    -----
    Macro averaging drops datasets whose metric is missing or ``nan``, so a run
    in which only one external dataset evaluates still yields a number -- one
    computed from a single dataset. The targets themselves are project KPIs, not
    manuscript values.
    """
    crispr = external_per_dataset.get("K562_CRISPRi", {})
    dual = external_per_dataset.get("K562_dualKO", {})

    def _macro_at(k: int) -> float:
        """Mean of the two external datasets' precision@k, ignoring ``nan``."""
        col = f"precision_at_{int(k)}"
        vals = [
            float(crispr.get(col, float("nan"))),
            float(dual.get(col, float("nan"))),
        ]
        vals = [x for x in vals if x == x]
        return float(np.mean(vals)) if vals else float("nan")

    def _ok(x: float, tgt: float) -> bool:
        """``nan``-safe ``x >= tgt`` test with the threshold passed explicitly."""
        return x == x and x >= float(tgt)

    med_auroc = float(test_auroc_median) if test_auroc_median is not None else float("nan")
    macro_p10 = _macro_at(10)
    macro_p30 = _macro_at(30)
    macro_p50 = _macro_at(50)

    ok_auroc = _ok(med_auroc, target_test_median_auroc)
    ok_p10 = _ok(macro_p10, target_macro_precision_at_10)
    ok_p30 = _ok(macro_p30, target_macro_precision_at_30)
    ok_p50 = _ok(macro_p50, target_macro_precision_at_50)

    return {
        "target_test_median_auroc": float(target_test_median_auroc),
        "target_macro_precision_at_10": float(target_macro_precision_at_10),
        "target_macro_precision_at_30": float(target_macro_precision_at_30),
        "target_macro_precision_at_50": float(target_macro_precision_at_50),
        "test_auroc_median": med_auroc if med_auroc == med_auroc else None,
        "external_macro_precision_at_10": macro_p10 if macro_p10 == macro_p10 else None,
        "external_macro_precision_at_30": macro_p30 if macro_p30 == macro_p30 else None,
        "external_macro_precision_at_50": macro_p50 if macro_p50 == macro_p50 else None,
        "test_median_auroc_ge_target": ok_auroc,
        "external_macro_precision_at_10_ge_target": ok_p10,
        "external_macro_precision_at_30_ge_target": ok_p30,
        "external_macro_precision_at_50_ge_target": ok_p50,
        "all_n4_gates_pass": bool(ok_auroc and ok_p10 and ok_p30 and ok_p50),
    }

