from __future__ import annotations

from typing import Iterable, Set, Tuple


def _edge_set(edges: Iterable[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    return {tuple(e) for e in edges}


def precision_recall_f1(pred_edges: Iterable[Tuple[str, str]], true_edges: Iterable[Tuple[str, str]]):
    p = _edge_set(pred_edges)
    t = _edge_set(true_edges)
    tp = len(p & t)
    fp = len(p - t)
    fn = len(t - p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def shd(pred_edges: Iterable[Tuple[str, str]], true_edges: Iterable[Tuple[str, str]]) -> int:
    p = _edge_set(pred_edges)
    t = _edge_set(true_edges)
    return len(p.symmetric_difference(t))

