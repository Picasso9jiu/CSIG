"""Frozen feature, target-matching, and seed-mapping helpers for M108."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment

from utils.temporal_memory_inference import _frame_tensor


def extract_frozen_h4_features_and_scores(
    model,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
    feature_downsample=4,
):
    """Return M26 event scores and frozen H/4 decoder features for one video."""
    if bool(getattr(model, 'confidence_head_enabled', False)):
        raise ValueError('M108 requires the released M26 event path.')
    context_bins = int(context_bins)
    width = int(width)
    height = int(height)
    inference_batch_size = int(inference_batch_size)
    feature_downsample = int(feature_downsample)
    if inference_batch_size <= 0 or feature_downsample <= 0:
        raise ValueError('inference_batch_size and feature_downsample must be positive.')
    bin_count = len(video.event_indices_by_bin)
    if bin_count <= 0:
        raise ValueError('video must contain temporal bins.')
    local_context = bool(getattr(model, 'local_temporal_context_enabled', False))
    local_kernel = int(getattr(model, 'local_temporal_context_kernel_size', 11))
    bottlenecks = []
    with torch.no_grad():
        for start in range(0, bin_count, inference_batch_size):
            bins = list(range(start, min(start + inference_batch_size, bin_count)))
            frames = _frame_tensor(
                video, bins, context_bins, width, height, log_count_clip, device,
                local_temporal_context_enabled=local_context,
                local_temporal_context_kernel_size=local_kernel,
            )
            bottlenecks.append(model.encode_bottleneck(frames))
        residuals = model.temporal_residual(torch.cat(bottlenecks, dim=0))
        scores = np.empty(video.locations.shape[0], dtype=np.float32)
        features = []
        for start in range(0, bin_count, inference_batch_size):
            bins = list(range(start, min(start + inference_batch_size, bin_count)))
            frames = _frame_tensor(
                video, bins, context_bins, width, height, log_count_clip, device,
                local_temporal_context_enabled=local_context,
                local_temporal_context_kernel_size=local_kernel,
            )
            decoded = model.decode_features_with_residual(
                frames, residuals[start:start + len(bins)]
            )
            logits = model.base.head(decoded)
            probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            pooled = functional.avg_pool2d(
                decoded, feature_downsample, feature_downsample, ceil_mode=True,
            )
            features.append(pooled.cpu())
            for local_index, temporal_bin in enumerate(bins):
                indices = video.event_indices_by_bin[temporal_bin]
                if indices.size:
                    locations = video.locations[indices]
                    scores[indices] = probabilities[
                        local_index, locations[:, 1], locations[:, 0]
                    ]
    if not np.isfinite(scores).all():
        raise RuntimeError('M108 feature extraction produced non-finite scores.')
    return torch.from_numpy(scores), torch.cat(features, dim=0)


def track_targets_from_video(video, width, height):
    """Build one full-video target trajectory tensor from labelled training data."""
    width = float(width)
    height = float(height)
    labels = np.asarray(video.labels) > 0.5
    target_ids = np.asarray(video.target_ids, dtype=np.int64)
    valid = labels & (target_ids > 0)
    ids = np.unique(target_ids[valid])
    time_count = len(video.event_indices_by_bin)
    active = np.zeros((time_count, ids.size), dtype=np.float32)
    centers = np.zeros((time_count, ids.size, 2), dtype=np.float32)
    scales = np.zeros((time_count, ids.size), dtype=np.float32)
    if ids.size == 0:
        return {
            'target_ids': ids,
            'active': torch.from_numpy(active),
            'centers': torch.from_numpy(centers),
            'scales': torch.from_numpy(scales),
        }
    index_by_id = {int(target_id): index for index, target_id in enumerate(ids.tolist())}
    for temporal_bin, indices in enumerate(video.event_indices_by_bin):
        if not indices.size:
            continue
        local_ids = target_ids[indices]
        local_labels = labels[indices]
        locations = video.locations[indices]
        for target_id in np.unique(local_ids[local_labels & (local_ids > 0)]).tolist():
            target_index = index_by_id[int(target_id)]
            mask = local_labels & (local_ids == target_id)
            xy = locations[mask, :2].astype(np.float32, copy=False)
            centre = xy.mean(axis=0)
            radius = np.sqrt(((xy - centre[None, :]) ** 2).sum(axis=1))
            active[temporal_bin, target_index] = 1.0
            centers[temporal_bin, target_index] = (centre[0] / width, centre[1] / height)
            scales[temporal_bin, target_index] = max(
                float(np.percentile(radius, 90)) / max(width, height),
                1.0 / max(width, height),
            )
    return {
        'target_ids': ids,
        'active': torch.from_numpy(active),
        'centers': torch.from_numpy(centers),
        'scales': torch.from_numpy(scales),
    }


def _hungarian_assignment(outputs, targets):
    active = targets['active'].to(outputs['existence_logits'].device)
    centers = targets['centers'].to(outputs['centers'].device)
    scales = targets['scales'].to(outputs['scales'].device)
    query_count = outputs['existence_logits'].shape[1]
    target_count = active.shape[1]
    if target_count == 0:
        return [], active, centers, scales
    probability = torch.sigmoid(outputs['existence_logits']).clamp(1e-5, 1.0 - 1e-5)
    # [Q, K], evaluated across the entire video so a query keeps one identity.
    presence_cost = -(
        active[:, None, :] * torch.log(probability[:, :, None])
        + (1.0 - active[:, None, :]) * torch.log(1.0 - probability[:, :, None])
    ).mean(dim=0)
    centre_error = torch.abs(
        outputs['centers'][:, :, None, :] - centers[:, None, :, :]
    ).sum(dim=-1)
    centre_cost = (centre_error * active[:, None, :]).sum(dim=0) / active.sum(dim=0).clamp_min(1.0)
    scale_error = torch.abs(
        outputs['scales'][:, :, None] - scales[:, None, :]
    )
    scale_cost = (scale_error * active[:, None, :]).sum(dim=0) / active.sum(dim=0).clamp_min(1.0)
    cost = presence_cost + 4.0 * centre_cost + 0.5 * scale_cost
    query_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
    return list(zip(query_indices.tolist(), target_indices.tolist())), active, centers, scales


def track_query_loss(outputs, targets, no_object_reduction='group_mean'):
    """Track-level Hungarian loss with explicit unmatched-query no-object loss."""
    if no_object_reduction not in {'group_mean', 'per_query_weighted_mean'}:
        raise ValueError(
            "no_object_reduction must be 'group_mean' or "
            "'per_query_weighted_mean'."
        )
    assignments, active, centers, scales = _hungarian_assignment(outputs, targets)
    logits = outputs['existence_logits']
    query_count = logits.shape[1]
    losses = []
    assigned_queries = set()
    for query_index, target_index in assignments:
        assigned_queries.add(query_index)
        target_active = active[:, target_index]
        positive_weight = torch.tensor(3.0, device=logits.device, dtype=logits.dtype)
        presence = functional.binary_cross_entropy_with_logits(
            logits[:, query_index], target_active, pos_weight=positive_weight,
        )
        active_mask = target_active > 0.5
        centre = functional.l1_loss(
            outputs['centers'][active_mask, query_index],
            centers[active_mask, target_index],
        ) if bool(active_mask.any()) else logits.sum() * 0.0
        scale = functional.l1_loss(
            outputs['scales'][active_mask, query_index],
            scales[active_mask, target_index],
        ) if bool(active_mask.any()) else logits.sum() * 0.0
        pair_mask = active_mask[1:] & active_mask[:-1]
        velocity = logits.sum() * 0.0
        if bool(pair_mask.any()):
            target_velocity = centers[1:, target_index] - centers[:-1, target_index]
            velocity = functional.l1_loss(
                outputs['velocities'][1:, query_index][pair_mask],
                target_velocity[pair_mask],
            )
        losses.append(presence + 4.0 * centre + 0.5 * scale + velocity)
    unmatched = sorted(set(range(query_count)) - assigned_queries)
    if unmatched:
        if no_object_reduction == 'group_mean':
            no_object = functional.binary_cross_entropy_with_logits(
                logits[:, unmatched], torch.zeros_like(logits[:, unmatched]),
            )
            losses.append(0.25 * no_object)
        else:
            no_object = functional.binary_cross_entropy_with_logits(
                logits[:, unmatched], torch.zeros_like(logits[:, unmatched]),
                reduction='none',
            ).mean(dim=0)
            weighted_sum = (
                torch.stack(losses).sum() if losses else logits.sum() * 0.0
            )
            weighted_sum = weighted_sum + 0.25 * no_object.sum()
            normalizer = len(losses) + 0.25 * len(unmatched)
            return weighted_sum / normalizer, {
                'matched_tracks': len(assignments),
                'unmatched_queries': len(unmatched),
                'no_object_reduction': no_object_reduction,
            }
    if not losses:
        return logits.sum() * 0.0, {
            'matched_tracks': 0,
            'unmatched_queries': query_count,
            'no_object_reduction': no_object_reduction,
        }
    return torch.stack(losses).mean(), {
        'matched_tracks': len(assignments),
        'unmatched_queries': len(unmatched),
        'no_object_reduction': no_object_reduction,
    }


def track_query_seed_indices(
    outputs,
    video,
    raw_scores,
    base_mask,
    width,
    height,
    presence_threshold=0.50,
    minimum_raw_score=None,
):
    """Map each accepted query/bin to at most one currently-negative event."""
    raw_scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)
    base_mask = np.asarray(base_mask, dtype=bool).reshape(-1)
    if raw_scores.size != base_mask.size or raw_scores.size != video.locations.shape[0]:
        raise ValueError('raw_scores, base_mask, and video events must align.')
    if minimum_raw_score is not None:
        minimum_raw_score = float(minimum_raw_score)
        if not 0.0 <= minimum_raw_score <= 1.0:
            raise ValueError('minimum_raw_score must be in [0, 1].')
    probability = torch.sigmoid(outputs['existence_logits']).detach().cpu().numpy()
    centres = outputs['centers'].detach().cpu().numpy()
    scales = outputs['scales'].detach().cpu().numpy()
    selected = []
    width = float(width)
    height = float(height)
    for temporal_bin, indices in enumerate(video.event_indices_by_bin):
        if not indices.size:
            continue
        used = set()
        query_order = np.argsort(-probability[temporal_bin], kind='stable')
        for query_index in query_order.tolist():
            if probability[temporal_bin, query_index] < float(presence_threshold):
                continue
            point = centres[temporal_bin, query_index] * np.asarray([width, height])
            # A learned target radius is bounded by a fixed 3--6 px deployment range.
            radius = float(np.clip(scales[temporal_bin, query_index] * max(width, height) * 2.0, 3.0, 6.0))
            locations = video.locations[indices, :2].astype(np.float32, copy=False)
            distances = np.linalg.norm(locations - point[None, :], axis=1)
            valid = (~base_mask[indices]) & (distances <= radius)
            if minimum_raw_score is not None:
                valid &= raw_scores[indices] >= minimum_raw_score
            if not valid.any():
                continue
            candidate_indices = indices[valid]
            candidate_distances = distances[valid]
            nonboundary = np.remainder(video.locations[candidate_indices, 2], 50) != 0
            pool = np.flatnonzero(nonboundary)
            if pool.size == 0:
                pool = np.arange(candidate_indices.size)
            # Prefer existing high M26/P41 evidence, then spatial proximity, then index.
            ordered = sorted(
                pool.tolist(),
                key=lambda local: (
                    -float(raw_scores[candidate_indices[local]]),
                    float(candidate_distances[local]),
                    int(candidate_indices[local]),
                ),
            )
            chosen = next(
                (int(candidate_indices[local]) for local in ordered if int(candidate_indices[local]) not in used),
                None,
            )
            if chosen is not None:
                used.add(chosen)
                selected.append(chosen)
    return np.asarray(sorted(set(selected)), dtype=np.int64)
