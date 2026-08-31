"""Exact Challenge 2 ROC-boundary helpers for postprocessing experiments.

``evalute.roc_update`` scores only the union of its strict 50-unit temporal
intervals.  This module exposes the same membership rule without changing
event timestamps or mutating already-positive predictions.
"""

import torch


def _integer_timestamps(timestamps):
    timestamps = torch.as_tensor(timestamps).reshape(-1)
    if timestamps.numel() == 0:
        raise ValueError('timestamps must not be empty.')
    if timestamps.is_floating_point():
        rounded = torch.round(timestamps)
        if not torch.equal(timestamps, rounded):
            raise ValueError(
                'Floating timestamps must be exact integer representations.'
            )
        timestamps = rounded.to(dtype=torch.long)
    elif timestamps.dtype == torch.bool:
        raise ValueError('timestamps must use an integer numeric dtype.')
    else:
        timestamps = timestamps.to(dtype=torch.long)
    return timestamps


def roc_included_mask_reference(timestamps, pd_detT=50):
    """Literal vectorized transcription of ``evalute.roc_update`` windows."""
    timestamps = _integer_timestamps(timestamps)
    pd_detT = int(pd_detT)
    if pd_detT <= 0:
        raise ValueError('pd_detT must be positive.')
    window_count = int(
        (timestamps.max().item() - timestamps.min().item()) / pd_detT + 1
    )
    included = torch.zeros_like(timestamps, dtype=torch.bool)
    for window_index in range(window_count):
        included |= (
            (timestamps > window_index * pd_detT)
            & (timestamps < (window_index + 1) * pd_detT)
        )
    return included


def build_exact_roc_excluded_mask(timestamps, pd_detT=50):
    """Return events excluded from ``evalute.roc_update``.

    The closed-form branch is equivalent to the evaluator's union of strict
    intervals for integer timestamps.  ``roc_included_mask_reference`` remains
    available for parity tests against the literal evaluator expression.
    """
    timestamps = _integer_timestamps(timestamps)
    pd_detT = int(pd_detT)
    if pd_detT <= 0:
        raise ValueError('pd_detT must be positive.')
    window_count = int(
        (timestamps.max().item() - timestamps.min().item()) / pd_detT + 1
    )
    included = (
        (timestamps > 0)
        & (timestamps < window_count * pd_detT)
        & (torch.remainder(timestamps, pd_detT) != 0)
    )
    return ~included


def recover_roc_excluded_events(
    base_mask,
    raw_scores,
    timestamps,
    decision_threshold,
    delta=0.0,
    pd_detT=50,
):
    """Monotonically add only high-score ROC-excluded negative events.

    ``base_mask`` must already include P0/P0c/P18.  ``raw_scores`` is the
    production P41 score vector from before those postprocessors.  The result
    is a binary mask and an ``extra`` mask; neither input is mutated.
    """
    base_mask = torch.as_tensor(base_mask).reshape(-1).to(dtype=torch.bool)
    raw_scores = torch.as_tensor(raw_scores).reshape(-1)
    timestamps = _integer_timestamps(timestamps)
    if not (
        base_mask.numel() == raw_scores.numel() == timestamps.numel()
    ):
        raise ValueError('base_mask, raw_scores, and timestamps must align.')
    decision_threshold = float(decision_threshold)
    delta = float(delta)
    if not 0.0 <= decision_threshold <= 1.0:
        raise ValueError('decision_threshold must be in [0, 1].')
    if delta < 0.0:
        raise ValueError('delta must be non-negative.')

    excluded = build_exact_roc_excluded_mask(timestamps, pd_detT=pd_detT)
    extra = (
        excluded
        & ~base_mask
        & (raw_scores >= decision_threshold - delta)
    )
    return base_mask | extra, extra
