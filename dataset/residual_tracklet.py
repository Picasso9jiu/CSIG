"""Frozen M75 residual-tracklet cache utilities.

The candidate cache is built from inference-visible information only.  This
module turns that immutable cache into graph tensors for M75-C.  Labels and
target IDs are returned separately for training/evaluation and never form an
input feature.
"""

from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import torch


TIME_BIN_SIZE = 50
LOW_DENSITY_CUTOFF = 30000
FRAME_WIDTH = 346
FRAME_HEIGHT = 260
PATCH_RADIUS = 7
PATCH_CONTEXT_BINS = 5
PATCH_LOG_COUNT_CLIP = 4.0
FEATURE_NAMES = (
    'raw_score',
    'score_margin',
    'route_m10',
    'route_m26',
    'log_event_count',
    'normalized_nms_rank',
    'was_threshold_positive_p0_removed',
    'local_count_3',
    'local_count_7',
    'local_count_11',
    'occupancy_11',
    'polarity_balance_11',
    'center_to_local_count_ratio',
    'boundary_distance',
)
EDGE_FEATURE_DIM = 6


def _window_labels(record):
    labels = record['seg_label'].detach().cpu().numpy() > 0.5
    target_ids = np.asarray(record['idx_label'])
    return labels, target_ids


def _local_observable_features(locs, polarities, xy, bins):
    """Use only events and polarities to construct fixed local scalar features."""
    coordinates = locs.detach().cpu().numpy()[:, 1:4].astype(np.int64, copy=False)
    polarities = np.asarray(polarities, dtype=np.float64)
    event_bins = coordinates[:, 2] // TIME_BIN_SIZE
    output = np.zeros((len(xy), 7), dtype=np.float64)
    by_bin = defaultdict(list)
    for event_index, time_bin in enumerate(event_bins.tolist()):
        by_bin[int(time_bin)].append(event_index)
    for time_bin, event_indices_list in by_bin.items():
        node_indices = np.flatnonzero(bins == time_bin)
        if node_indices.size == 0:
            continue
        event_indices = np.asarray(event_indices_list, dtype=np.int64)
        points = coordinates[event_indices, :2]
        values = polarities[event_indices] > 0.5
        count_image = np.zeros((FRAME_HEIGHT + 1, FRAME_WIDTH + 1), dtype=np.int32)
        positive_image = np.zeros_like(count_image)
        np.add.at(count_image, (points[:, 1] + 1, points[:, 0] + 1), 1)
        np.add.at(positive_image, (points[:, 1] + 1, points[:, 0] + 1), values.astype(np.int32))
        integral = count_image.cumsum(axis=0).cumsum(axis=1)
        positive_integral = positive_image.cumsum(axis=0).cumsum(axis=1)
        occupied = (count_image > 0).astype(np.int32).cumsum(axis=0).cumsum(axis=1)
        x = xy[node_indices, 0].astype(np.int64)
        y = xy[node_indices, 1].astype(np.int64)
        windows = []
        for radius in (1, 3, 5):
            x0 = np.maximum(x - radius, 0)
            y0 = np.maximum(y - radius, 0)
            x1 = np.minimum(x + radius, FRAME_WIDTH - 1)
            y1 = np.minimum(y + radius, FRAME_HEIGHT - 1)
            count = (
                integral[y1 + 1, x1 + 1] - integral[y0, x1 + 1]
                - integral[y1 + 1, x0] + integral[y0, x0]
            )
            windows.append(count.astype(np.float64))
            if radius == 5:
                positive = (
                    positive_integral[y1 + 1, x1 + 1] - positive_integral[y0, x1 + 1]
                    - positive_integral[y1 + 1, x0] + positive_integral[y0, x0]
                )
                occupied_count = (
                    occupied[y1 + 1, x1 + 1] - occupied[y0, x1 + 1]
                    - occupied[y1 + 1, x0] + occupied[y0, x0]
                )
                area = (x1 - x0 + 1) * (y1 - y0 + 1)
                windows.extend((positive.astype(np.float64), occupied_count / area))
        count3, count7, count11, positive11, occupancy11 = windows
        output[node_indices] = np.column_stack((
            np.log1p(count3),
            np.log1p(count7),
            np.log1p(count11),
            occupancy11,
            (2.0 * positive11 - count11) / np.maximum(count11, 1.0),
            count3 / np.maximum(count11, 1.0),
            np.minimum(np.minimum(x, FRAME_WIDTH - 1 - x), np.minimum(y, FRAME_HEIGHT - 1 - y))
            / float(min(FRAME_WIDTH - 1, FRAME_HEIGHT - 1)),
        ))
    return output


def _patches(locs, polarities, xy, bins):
    """Build 5 x 2 polarity-count 15x15 patches with zero-padded boundaries."""
    coordinates = locs.detach().cpu().numpy()[:, 1:4].astype(np.int64, copy=False)
    polarities = np.asarray(polarities, dtype=np.float64) > 0.5
    event_bins = coordinates[:, 2] // TIME_BIN_SIZE
    node_count = len(xy)
    patches = np.zeros((node_count, PATCH_CONTEXT_BINS * 2, PATCH_RADIUS * 2 + 1, PATCH_RADIUS * 2 + 1), dtype=np.float32)
    by_bin = defaultdict(list)
    for event_index, time_bin in enumerate(event_bins.tolist()):
        by_bin[int(time_bin)].append(event_index)
    node_by_bin = defaultdict(list)
    for node, time_bin in enumerate(bins.tolist()):
        node_by_bin[int(time_bin)].append(node)
    for source_bin, event_indices_list in by_bin.items():
        event_indices = np.asarray(event_indices_list, dtype=np.int64)
        points = coordinates[event_indices, :2]
        values = polarities[event_indices].astype(np.int64)
        for relative in range(-2, 3):
            target_bin = source_bin - relative
            for node in node_by_bin.get(target_bin, ()):
                dx = points[:, 0] - int(xy[node, 0])
                dy = points[:, 1] - int(xy[node, 1])
                inside = (np.abs(dx) <= PATCH_RADIUS) & (np.abs(dy) <= PATCH_RADIUS)
                if not inside.any():
                    continue
                channel = (relative + 2) * 2 + values[inside]
                np.add.at(
                    patches[node],
                    (channel, dy[inside] + PATCH_RADIUS, dx[inside] + PATCH_RADIUS),
                    1.0,
                )
    np.log1p(patches, out=patches)
    np.minimum(patches, PATCH_LOG_COUNT_CLIP, out=patches)
    patches /= PATCH_LOG_COUNT_CLIP
    return patches


def _edge_features(candidate, local_count_11):
    xy = np.asarray(candidate['xy'], dtype=np.float64)
    bins = np.asarray(candidate['time_bins'], dtype=np.int64)
    scores = np.asarray(candidate['scores'], dtype=np.float64)
    edges = np.asarray(candidate['graph_edges'], dtype=np.int64).reshape(-1, 3)
    features = np.zeros((len(edges), EDGE_FEATURE_DIM), dtype=np.float32)
    for edge_index, (source, destination, gap) in enumerate(edges.tolist()):
        radius = 24.0 if int(gap) == 1 else 40.0
        delta = xy[destination] - xy[source]
        distance = float(np.linalg.norm(delta))
        features[edge_index] = (
            delta[0] / radius,
            delta[1] / radius,
            distance / radius,
            float(gap),
            scores[destination] - scores[source],
            local_count_11[destination] - local_count_11[source],
        )
    return features


def build_tracklet_record(score_record, candidate_record, polarities):
    """Create one M75-C graph record.  Inputs exclude labels and target IDs."""
    candidate = candidate_record['candidates']
    node_event_indices = np.asarray(candidate['event_indices'], dtype=np.int64)
    scores = np.asarray(candidate['scores'], dtype=np.float64)
    xy = np.asarray(candidate['xy'], dtype=np.int64)
    bins = np.asarray(candidate['time_bins'], dtype=np.int64)
    ranks = np.asarray(candidate['nms_rank'], dtype=np.float64)
    p0_removed = np.asarray(candidate['p0_removed'], dtype=np.float64)
    event_count = int(score_record['event_count'])
    threshold = float(candidate_record['threshold'])
    observable = _local_observable_features(score_record['locs'], polarities, xy, bins)
    route_m10 = float(event_count <= LOW_DENSITY_CUTOFF)
    max_rank = {
        int(time_bin): max(ranks[bins == time_bin]) for time_bin in np.unique(bins)
    }
    scalar = np.column_stack((
        scores,
        scores - threshold,
        np.full(len(scores), route_m10),
        np.full(len(scores), 1.0 - route_m10),
        np.full(len(scores), math.log1p(event_count)),
        np.asarray([ranks[index] / max_rank[int(bins[index])] for index in range(len(ranks))]),
        p0_removed,
        observable,
    )).astype(np.float32)
    patches = _patches(score_record['locs'], polarities, xy, bins)
    edges = np.asarray(candidate['graph_edges'], dtype=np.int64).reshape(-1, 3)
    edge_features = _edge_features(candidate, observable[:, 2])
    labels, target_ids = _window_labels(score_record)
    node_labels = labels[node_event_indices].astype(np.float32)
    node_target_ids = target_ids[node_event_indices].astype(np.int64)
    return {
        'patches': torch.from_numpy(patches),
        'scalar_features': torch.from_numpy(scalar),
        'edge_index': torch.from_numpy(edges[:, :2].T.copy()).to(torch.int64),
        'edge_features': torch.from_numpy(edge_features),
        'node_event_indices': torch.from_numpy(node_event_indices),
        'node_time_bins': torch.from_numpy(bins),
        'production_mask': torch.from_numpy(np.asarray(candidate_record['production_mask'], dtype=bool)),
        # Targets are training/evaluation-only and must not be passed to model.forward.
        'node_labels': torch.from_numpy(node_labels),
        'node_target_ids': torch.from_numpy(node_target_ids),
        'event_count': event_count,
    }


def assert_model_input(record):
    """Reject accidental propagation of identity or supervision to model input."""
    allowed = {'patches', 'scalar_features', 'edge_index', 'edge_features'}
    if set(record) & {'name', 'file_name', 'video_index', 'validation_index'}:
        raise ValueError('M75 model input must not contain video identity.')
    return {key: record[key] for key in allowed}
