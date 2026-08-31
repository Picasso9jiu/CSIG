"""Inference helpers for M95's frozen centre proposer."""

import numpy as np
import torch
import torch.nn.functional as F

from utils.temporal_memory_inference import _frame_tensor


def predict_frozen_center_heatmaps(
    model,
    proposer,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
    feature_downsample=4,
):
    """Return one sigmoid centre heatmap per metric-time bin.

    The forward is label-free: ``video`` is consumed only through observable
    event coordinates, polarity and timestamps.  The frozen M26 model is
    evaluated with its released full-stream memory pass; the proposer sees
    only ``decoded0`` features after that pass.
    """
    if not model.training and not proposer.training:
        pass
    context_bins = int(context_bins)
    width = int(width)
    height = int(height)
    inference_batch_size = int(inference_batch_size)
    feature_downsample = int(feature_downsample)
    if inference_batch_size <= 0 or feature_downsample <= 0:
        raise ValueError('inference_batch_size and feature_downsample must be positive.')
    temporal_bin_count = len(video.event_indices_by_bin)
    if temporal_bin_count <= 0:
        raise ValueError('video must contain temporal bins.')

    model_local_context = bool(getattr(model, 'local_temporal_context_enabled', False))
    local_context_kernel = int(
        getattr(model, 'local_temporal_context_kernel_size', 11)
    )
    fine_enabled = bool(getattr(model, 'fine_temporal_memory_enabled', False))
    bottlenecks = []
    fine_level2 = []
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            bins = list(range(start, min(start + inference_batch_size, temporal_bin_count)))
            frames = _frame_tensor(
                video,
                bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
                local_temporal_context_enabled=model_local_context,
                local_temporal_context_kernel_size=local_context_kernel,
            )
            if fine_enabled:
                level2, bottleneck = model.encode_memory_features(frames)
                fine_level2.append(level2)
            else:
                bottleneck = model.encode_bottleneck(frames)
            bottlenecks.append(bottleneck)
        bottlenecks = torch.cat(bottlenecks, dim=0)
        residuals = model.temporal_residual(bottlenecks)
        fine_residuals = None
        if fine_enabled:
            fine_residuals = model.fine_temporal_residual(torch.cat(fine_level2, dim=0))

        heatmaps = []
        for start in range(0, temporal_bin_count, inference_batch_size):
            bins = list(range(start, min(start + inference_batch_size, temporal_bin_count)))
            frames = _frame_tensor(
                video,
                bins,
                context_bins,
                width,
                height,
                log_count_clip,
                device,
                local_temporal_context_enabled=model_local_context,
                local_temporal_context_kernel_size=local_context_kernel,
            )
            decoded = model.decode_features_with_residual(
                frames,
                residuals[start:start + len(bins)],
                fine_residual=(
                    fine_residuals[start:start + len(bins)]
                    if fine_residuals is not None else None
                ),
            )
            pooled = F.avg_pool2d(
                decoded,
                kernel_size=feature_downsample,
                stride=feature_downsample,
                ceil_mode=True,
            )
            heatmaps.append(torch.sigmoid(proposer(pooled)).squeeze(1).cpu())
    return torch.cat(heatmaps, dim=0).numpy().astype(np.float32, copy=False)


def predict_frozen_center_scores_and_heatmaps(
    model,
    proposer,
    video,
    device,
    context_bins,
    width,
    height,
    inference_batch_size,
    log_count_clip=4.0,
    feature_downsample=4,
):
    """Compute released M26 scores and M95 heatmaps in one raw-grid pass.

    M95 is specified only for the frozen M26 checkpoint, whose event path is
    exactly ``base.head(decoded0)``.  Reject checkpoints with output branches
    that would make that equivalence false instead of silently changing the
    production score path.
    """
    if bool(getattr(model, 'confidence_head_enabled', False)):
        raise ValueError('M95 does not support a confidence-head M26 checkpoint.')
    if bool(getattr(model.base, 'target_center_enabled', False)):
        raise ValueError('M95 does not support a target-centre M26 checkpoint.')
    if bool(getattr(model, 'center_memory_enabled', False)):
        raise ValueError('M95 does not support a centre-memory M26 checkpoint.')

    context_bins = int(context_bins)
    width = int(width)
    height = int(height)
    inference_batch_size = int(inference_batch_size)
    feature_downsample = int(feature_downsample)
    if inference_batch_size <= 0 or feature_downsample <= 0:
        raise ValueError('inference_batch_size and feature_downsample must be positive.')
    temporal_bin_count = len(video.event_indices_by_bin)
    if temporal_bin_count <= 0:
        raise ValueError('video must contain temporal bins.')

    model_local_context = bool(getattr(model, 'local_temporal_context_enabled', False))
    local_context_kernel = int(
        getattr(model, 'local_temporal_context_kernel_size', 11)
    )
    fine_enabled = bool(getattr(model, 'fine_temporal_memory_enabled', False))
    bottlenecks = []
    fine_level2 = []
    with torch.no_grad():
        for start in range(0, temporal_bin_count, inference_batch_size):
            bins = list(range(start, min(start + inference_batch_size, temporal_bin_count)))
            frames = _frame_tensor(
                video, bins, context_bins, width, height, log_count_clip, device,
                local_temporal_context_enabled=model_local_context,
                local_temporal_context_kernel_size=local_context_kernel,
            )
            if fine_enabled:
                level2, bottleneck = model.encode_memory_features(frames)
                fine_level2.append(level2)
            else:
                bottleneck = model.encode_bottleneck(frames)
            bottlenecks.append(bottleneck)
        residuals = model.temporal_residual(torch.cat(bottlenecks, dim=0))
        fine_residuals = None
        if fine_enabled:
            fine_residuals = model.fine_temporal_residual(torch.cat(fine_level2, dim=0))

        scores = np.empty(video.locations.shape[0], dtype=np.float32)
        heatmaps = []
        for start in range(0, temporal_bin_count, inference_batch_size):
            bins = list(range(start, min(start + inference_batch_size, temporal_bin_count)))
            frames = _frame_tensor(
                video, bins, context_bins, width, height, log_count_clip, device,
                local_temporal_context_enabled=model_local_context,
                local_temporal_context_kernel_size=local_context_kernel,
            )
            decoded = model.decode_features_with_residual(
                frames,
                residuals[start:start + len(bins)],
                fine_residual=(
                    fine_residuals[start:start + len(bins)]
                    if fine_residuals is not None else None
                ),
            )
            logits = model.base.head(decoded)
            probabilities = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            pooled = F.avg_pool2d(
                decoded,
                kernel_size=feature_downsample,
                stride=feature_downsample,
                ceil_mode=True,
            )
            heatmaps.append(torch.sigmoid(proposer(pooled)).squeeze(1).cpu())
            for local_index, temporal_bin in enumerate(bins):
                event_indices = video.event_indices_by_bin[temporal_bin]
                if event_indices.size == 0:
                    continue
                locations = video.locations[event_indices]
                scores[event_indices] = probabilities[
                    local_index, locations[:, 1], locations[:, 0]
                ]
    if not np.isfinite(scores).all():
        raise RuntimeError('M95 joint inference produced non-finite event scores.')
    return torch.from_numpy(scores), torch.cat(heatmaps, dim=0).numpy().astype(
        np.float32, copy=False
    )


def local_peak_indices(heatmap, threshold, max_peaks, nms_radius=2):
    """Return deterministic local maxima from one 2-D heatmap."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError('heatmap must be two-dimensional.')
    threshold = float(threshold)
    max_peaks = int(max_peaks)
    nms_radius = int(nms_radius)
    if max_peaks <= 0 or nms_radius < 0:
        return []
    candidates = np.argwhere(heatmap >= threshold)
    if candidates.size == 0:
        return []
    values = heatmap[candidates[:, 0], candidates[:, 1]]
    order = np.lexsort((candidates[:, 1], candidates[:, 0], -values))
    selected = []
    for index in order.tolist():
        y, x = candidates[index].tolist()
        if any(
            abs(y - previous_y) <= nms_radius
            and abs(x - previous_x) <= nms_radius
            for previous_y, previous_x, _ in selected
        ):
            continue
        selected.append((y, x, float(values[index])))
        if len(selected) >= max_peaks:
            break
    return selected


def proposer_event_candidates(
    heatmaps,
    video,
    threshold,
    max_peaks=5,
    nms_radius=2,
    feature_downsample=4,
    max_event_distance=6.0,
):
    """Map heatmap peaks to at most one original event per peak and bin."""
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.ndim != 3:
        raise ValueError('heatmaps must have shape [T, H, W].')
    if heatmaps.shape[0] != len(video.event_indices_by_bin):
        raise ValueError('heatmap time dimension does not match video bins.')
    feature_downsample = float(feature_downsample)
    max_event_distance = float(max_event_distance)
    candidate_indices = []
    candidate_scores = []
    for temporal_bin, event_indices in enumerate(video.event_indices_by_bin):
        if event_indices.size == 0:
            continue
        peaks = local_peak_indices(
            heatmaps[temporal_bin], threshold, max_peaks, nms_radius
        )
        if not peaks:
            continue
        locations = np.asarray(video.locations[event_indices], dtype=np.float32)
        used = set()
        for peak_y, peak_x, peak_score in peaks:
            target_xy = np.asarray(
                [peak_x * feature_downsample, peak_y * feature_downsample],
                dtype=np.float32,
            )
            distances = np.linalg.norm(locations[:, :2] - target_xy[None, :], axis=1)
            order = np.argsort(distances, kind='stable')
            chosen = None
            for local_index in order.tolist():
                event_index = int(event_indices[local_index])
                if event_index in used:
                    continue
                if float(distances[local_index]) <= max_event_distance:
                    chosen = event_index
                    break
            if chosen is None:
                continue
            used.add(chosen)
            candidate_indices.append(chosen)
            candidate_scores.append(float(peak_score))
    if not candidate_indices:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    return (
        np.asarray(candidate_indices, dtype=np.int64),
        np.asarray(candidate_scores, dtype=np.float32),
    )
