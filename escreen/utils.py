import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import re
from typing import Any, Sequence
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import spearmanr

class boost_sampler():
    
    def __init__(self, labels, smoothing=1.0, temperature=0.7, seed=114514):
        """
        初始化采样器，预计算 平滑概率分布。
        
        Args:
            labels: 训练集的标签列表或 array。
            smoothing: 拉普拉斯平滑系数，建议 1.0。
            temperature: 温度系数，0.5-0.7 可有效防止过拟合。
            seed: 随机种子，确保实验可重复性。
        """
        self.labels = np.array(labels)
        self.smoothing = smoothing
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)  # 使用更安全的 numpy 随机生成器
        
        # 预计算采样概率，避免每个 Epoch 重复计算
        self.prob_dist = self._precompute_probabilities()

    def _precompute_probabilities(self):
        n_total = len(self.labels)
        classes, counts = np.unique(self.labels, return_counts=True)
        
        # 1. 计算平滑权重: w = N / (count + alpha)
        smoothed_counts = counts + self.smoothing
        class_weights_raw = n_total / smoothed_counts
        
        # 2. 温度缩放: w = w^T
        # T < 1.0 会压缩高权重与低权重之间的差距
        class_weights_map = {
            cls: np.power(class_weights_raw[i], self.temperature) 
            for i, cls in enumerate(classes)
        }
        
        # 3. 映射并归一化
        sample_weights = np.array([class_weights_map[l] for l in self.labels])
        return sample_weights / np.sum(sample_weights)

    def get(self, num_samples=None):
        """
        在每个 Epoch 调用，获取重采样后的索引。
        
        Args:
            num_samples: 采样数量，默认为数据集全长。
        """
        n_total = len(self.labels)
        size = num_samples if num_samples is not None else n_total
        
        # 执行带放回抽样 (Replacement=True 是处理不平衡的核心)
        indices = self.rng.choice(
            np.arange(n_total), 
            size=size, 
            replace=True, 
            p=self.prob_dist
        )
        return indices


def auroc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    return float(average_precision_score(y_true, y_score))


def spearman_group(pred: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
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
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return float("nan")
    k = min(int(k), int(y_true.size))
    order = np.argsort(-y_score)
    top = y_true[order[:k]]
    return float(np.sum(top) / n_pos)


def ranking_metrics_minimal(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """AUROC and AUPRC only (no top-k or label rate)."""
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

    def _spearman(model, val_data: dict, device: str) -> float:
        preds, y_reg = model.predict(val_data, batch_size=256, device=device, verbose=False, with_true=True)
        return float(spearman_group(preds, y_reg, val_data["cell_type"]))

    def _macro_auroc(model, val_data: dict, device: str) -> float:
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

    def _combo(model, val_data: dict, device: str) -> float:
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
        preds, _y = model.predict(val_data, batch_size=256, device=device, verbose=False, with_true=False)
        return float(
            median_auprc(
                val_data["label"],
                preds,
                val_data["cell_line"],
                exclude_groups=exclude_groups,
            )
        )

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
    """Phase N1: maximize 721 valid median AUROC + K562 external CRISPRi/dualKO AUROC."""

    def _median_valid(model, _val_data: dict, device: str) -> float:
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
    """Boolean gates for Phase N1 KPIs (721 test median + K562 external AUROCs)."""
    k_crispr = float(external_per_dataset.get("K562_CRISPRi", {}).get("auroc", float("nan")))
    k_dual = float(external_per_dataset.get("K562_dualKO", {}).get("auroc", float("nan")))
    med = float(test_auroc_median)
    t = float(target)

    def _ok(x: float) -> bool:
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
    """Phase N2: 721 valid median AUPRC + K562 external AUPRC / precision@k (no external train)."""

    def _median_valid_auprc(model, _val_data: dict, device: str) -> float:
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
    """Phase N2 gates: 721 test median AUROC (>=0.70) + median AUPRC + K562 external AUPRC/p@k."""
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
        return x == x and x >= ta

    def _ok_pk(x: float) -> bool:
        if target_precision_at_k is None:
            return True
        return x == x and x >= float(target_precision_at_k)

    def _ok_auroc(x: float | None) -> bool:
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
    """Phase N3/N4: 721 valid median + K562 external macro precision@k."""

    def _median_valid_auprc(model, _val_data: dict, device: str) -> float:
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
        preds, _ = model.predict(
            batch, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        y = np.asarray(batch["label"], dtype=int)
        p = np.asarray(preds, dtype=float)
        return float(precision_at_k(y, p, int(k)))

    def _external_auroc(model, batch: dict, device: str) -> float:
        preds, _ = model.predict(
            batch, batch_size=predict_batch_size, device=device, verbose=False, with_true=False
        )
        y = np.asarray(batch["label"], dtype=int)
        p = np.asarray(preds, dtype=float)
        return float(auroc_safe(y, p))

    def _macro_external_pk(model, device: str, k: int) -> float:
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
    """CRISPRi ∩ dualKO loci; label=1 when both neg|score < neg_threshold."""
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
    """Phase N3 hPSC retrain: intersection golden @neg_threshold p@k early-stop."""

    def _median_valid_auprc(model, _val_data: dict, device: str) -> float:
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

    k_list = [int(x) for x in (earlystop_k_list or [5, 10, 15, 20, 25, 30])]
    default_w: dict[str, float] = {"median_auprc": 0.15, "golden_auroc": 0.0}
    for k in k_list:
        default_w[f"golden_precision_at_{k}"] = 0.85 / max(len(k_list), 1)
    w = combo_weights or default_w

    def _golden_intersection(model, _val_data: dict, device: str) -> float:
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
    """Phase N3: each external dataset p@10 and p@secondary_k must meet targets."""
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
        c_v = float(crispr.get(col, float("nan")))
        d_v = float(dual.get(col, float("nan")))
        vals = [x for x in (c_v, d_v) if x == x]
        return float(np.mean(vals)) if vals else float("nan")

    def _ok_pk(x: float, tgt: float) -> bool:
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


def compute_n6_per_dataset_gates(
    *,
    test_auroc_median: float | None = None,
    external_per_dataset: dict[str, dict[str, Any]],
    target_auroc: float = 0.7,
    target_p_at_k: dict[int, float] | None = None,
    baseline_test_median_auroc: float | None = 0.706,
    strict_pk_gt: bool = True,
) -> dict[str, Any]:
    """Per-dataset external AUROC + p@10/20/30/40 gates (selection-only tuning targets)."""
    pk_tgts = dict(target_p_at_k or N6_DEFAULT_PK_THRESHOLDS)

    def _dataset_checks(ds_name: str) -> dict[str, Any]:
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
    """Phase N6: external AUROC + per-dataset p@10/20/30/40 (dual weighted >= crispr)."""
    _ = valid_data, exclude_groups, earlystop_k_list
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
    """Phase N4: test median AUROC + external macro p@10/30/50 gates."""
    crispr = external_per_dataset.get("K562_CRISPRi", {})
    dual = external_per_dataset.get("K562_dualKO", {})

    def _macro_at(k: int) -> float:
        col = f"precision_at_{int(k)}"
        vals = [
            float(crispr.get(col, float("nan"))),
            float(dual.get(col, float("nan"))),
        ]
        vals = [x for x in vals if x == x]
        return float(np.mean(vals)) if vals else float("nan")

    def _ok(x: float, tgt: float) -> bool:
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

