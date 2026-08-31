"""Continuous full-frame sparse 3D clips for the M88 experiment.

The existing dataset classes intentionally sample a bounded number of events
from a whole video.  M88 uses a different contract: a short continuous clip
keeps every event in that interval, aggregates only exact 10-unit voxels, and
retains a map back to every original event.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import spconv.pytorch as spconv
from torch.utils.data import Dataset


def clip_start_grid(whole_t, clip_duration, stride):
    """Return half-open clip starts covering the complete time range."""
    whole_t = int(whole_t)
    clip_duration = int(clip_duration)
    stride = int(stride)
    if whole_t <= 0 or clip_duration <= 0 or clip_duration > whole_t:
        raise ValueError('clip_duration must be in (0, whole_t].')
    if stride <= 0:
        raise ValueError('stride must be positive.')
    last_start = whole_t - clip_duration
    starts = list(range(0, last_start + 1, stride))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return tuple(int(start) for start in starts)


def load_event_video(path, include_annotations=True):
    """Load an event video, optionally excluding non-observable annotations.

    ``include_annotations=False`` is for leakage-safe inference: the returned
    object contains only the event coordinates and polarity available at test
    time.  Training keeps the default and therefore retains labels/target ids.
    """
    with np.load(str(path)) as payload:
        locations = np.asarray(payload['ev_loc'], dtype=np.int64)
        normalized = np.asarray(payload['evs_norm'])
        polarity = np.rint(normalized[:, 3]).astype(np.int64, copy=True)
        if include_annotations:
            labels = np.asarray(normalized[:, 4], dtype=np.float32).copy()
            target_ids = np.rint(normalized[:, 5]).astype(np.int64, copy=True)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError('{} has invalid ev_loc shape.'.format(path))
    if len(locations) != len(polarity):
        raise ValueError('{} has inconsistent event fields.'.format(path))
    if len(locations) and not np.all(np.diff(locations[:, 2]) >= 0):
        order = np.argsort(locations[:, 2], kind='mergesort')
        locations = locations[order]
        polarity = polarity[order]
        if include_annotations:
            labels = labels[order]
            target_ids = target_ids[order]
    if np.any((polarity < 0) | (polarity > 1)):
        raise ValueError('{} contains polarity values outside {0,1}.'.format(path))
    video = {
        'locations': locations,
        'polarity': polarity,
    }
    if include_annotations:
        video['labels'] = labels
        video['target_ids'] = target_ids
    return video


def _aligned_clip_start(anchor_time, whole_t, clip_duration, rng, time_bin):
    """Place an anchor event at a random position inside a phase-aligned clip."""
    max_start = int(whole_t) - int(clip_duration)
    # Keep the start on the voxel grid while guaranteeing that an anchor with
    # arbitrary sub-bin phase remains inside the half-open clip.
    offset = int(rng.integers(0, int(clip_duration) // int(time_bin))) * int(time_bin)
    raw_start = int(anchor_time) - offset
    raw_start = min(max(raw_start, 0), max_start)
    return int((raw_start // int(time_bin)) * int(time_bin))


def build_sparse_clip(
    video,
    clip_start,
    clip_duration=400,
    temporal_bin=10,
    width=346,
    height=260,
    count_clip=8.0,
):
    """Build a complete-event sparse clip and its event-to-voxel map.

    Returns a dictionary with CPU NumPy arrays.  ``event_indices`` are indices
    into the (time-sorted) arrays returned by :func:`load_event_video`.
    """
    clip_start = int(clip_start)
    clip_duration = int(clip_duration)
    temporal_bin = int(temporal_bin)
    if clip_duration <= 0 or temporal_bin <= 0:
        raise ValueError('clip_duration and temporal_bin must be positive.')
    if clip_duration % temporal_bin:
        raise ValueError('clip_duration must be divisible by temporal_bin.')
    if count_clip <= 0:
        raise ValueError('count_clip must be positive.')
    locations = video['locations']
    event_count = int(locations.shape[0])
    left = int(np.searchsorted(locations[:, 2], clip_start, side='left'))
    right = int(np.searchsorted(
        locations[:, 2], clip_start + clip_duration, side='left'
    ))
    event_indices = np.arange(left, right, dtype=np.int64)
    local_locations = locations[left:right]
    time_bins = clip_duration // temporal_bin
    if local_locations.size:
        relative_time = (local_locations[:, 2] - clip_start) // temporal_bin
        if relative_time.min() < 0 or relative_time.max() >= time_bins:
            raise RuntimeError('relative time voxel is outside the clip.')
        raw_coords = np.column_stack((
            local_locations[:, 0],
            local_locations[:, 1],
            relative_time,
        )).astype(np.int64, copy=False)
        voxel_coords, event_to_voxel = np.unique(
            raw_coords,
            axis=0,
            return_inverse=True,
        )
        polarity = video['polarity'][left:right]
        voxel_features = np.zeros((len(voxel_coords), 3), dtype=np.float32)
        np.add.at(
            voxel_features[:, :2],
            (event_to_voxel, polarity),
            1.0,
        )
        phase = ((local_locations[:, 2] - clip_start) % temporal_bin).astype(
            np.float32,
            copy=False,
        ) / float(temporal_bin)
        phase_sum = np.zeros(len(voxel_coords), dtype=np.float32)
        np.add.at(phase_sum, event_to_voxel, phase)
        counts = np.bincount(event_to_voxel, minlength=len(voxel_coords)).astype(
            np.float32,
            copy=False,
        )
        voxel_features[:, 2] = phase_sum / np.maximum(counts, 1.0)
        np.minimum(voxel_features[:, :2], float(count_clip), out=voxel_features[:, :2])
        voxel_features[:, :2] = (
            np.log1p(voxel_features[:, :2]) / np.log1p(float(count_clip))
        )
        if 'labels' in video:
            labels = video['labels'][left:right].astype(np.float32, copy=True)
        else:
            labels = np.zeros(len(event_indices), dtype=np.float32)
        if 'target_ids' in video:
            target_ids = video['target_ids'][left:right].astype(np.int64, copy=True)
        else:
            target_ids = np.full(len(event_indices), -1, dtype=np.int64)
    else:
        voxel_coords = np.empty((0, 3), dtype=np.int64)
        event_to_voxel = np.empty((0,), dtype=np.int64)
        voxel_features = np.empty((0, 3), dtype=np.float32)
        labels = np.empty((0,), dtype=np.float32)
        target_ids = np.empty((0,), dtype=np.int64)
    if right > event_count:
        raise RuntimeError('clip event range exceeds source video.')
    return {
        'voxel_coords': voxel_coords,
        'voxel_features': voxel_features.astype(np.float32, copy=False),
        'event_to_voxel': event_to_voxel.astype(np.int64, copy=False),
        'event_indices': event_indices,
        'labels': labels,
        'target_ids': target_ids,
        'time_bins': ((local_locations[:, 2] - clip_start) // temporal_bin).astype(
            np.int64,
            copy=False,
        ) if local_locations.size else np.empty((0,), dtype=np.int64),
        'clip_start': clip_start,
        'spatial_shape': (int((width + 31) // 32 * 32), int((height + 31) // 32 * 32), time_bins),
    }


def sparse_tensor_from_clip(clip, device):
    """Create a batch-one spconv tensor and GPU event-alignment arrays."""
    voxel_coords = np.asarray(clip['voxel_coords'], dtype=np.int64)
    if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 3:
        raise ValueError('voxel_coords must have shape [V,3].')
    if len(voxel_coords) == 0:
        raise ValueError('a sparse clip must contain at least one event.')
    batch_column = np.zeros((len(voxel_coords), 1), dtype=np.int64)
    coords = np.concatenate((batch_column, voxel_coords), axis=1)
    features = torch.from_numpy(np.ascontiguousarray(clip['voxel_features'])).to(device)
    coords = torch.from_numpy(np.ascontiguousarray(coords)).to(device=device, dtype=torch.int32)
    sparse = spconv.SparseConvTensor(
        features,
        coords,
        list(clip['spatial_shape']),
        1,
    )
    event_to_voxel = torch.from_numpy(clip['event_to_voxel']).to(device=device, dtype=torch.long)
    labels = torch.from_numpy(clip['labels']).to(device=device, dtype=torch.float32)
    time_bins = torch.from_numpy(clip['time_bins']).to(device=device, dtype=torch.long)
    return sparse, event_to_voxel, labels, time_bins


class FullFrameSparse3DClipDataset(Dataset):
    """Sample continuous clips while retaining every event in each clip."""

    def __init__(
        self,
        root,
        video_names=None,
        whole_t=8000,
        clip_duration=400,
        temporal_bin=10,
        views_per_video=4,
        positive_clip_probability=0.75,
        random_seed=85,
        width=346,
        height=260,
        count_clip=8.0,
        cache_video_count=8,
    ):
        self.root = Path(root)
        self.whole_t = int(whole_t)
        self.clip_duration = int(clip_duration)
        self.temporal_bin = int(temporal_bin)
        self.views_per_video = int(views_per_video)
        self.positive_clip_probability = float(positive_clip_probability)
        self.random_seed = int(random_seed)
        self.width = int(width)
        self.height = int(height)
        self.count_clip = float(count_clip)
        self.cache_video_count = int(cache_video_count)
        self.current_epoch = 0
        if self.clip_duration % self.temporal_bin:
            raise ValueError('clip_duration must be divisible by temporal_bin.')
        if self.views_per_video <= 0:
            raise ValueError('views_per_video must be positive.')
        if not 0.0 <= self.positive_clip_probability <= 1.0:
            raise ValueError('positive_clip_probability must be in [0,1].')
        if self.cache_video_count <= 0:
            raise ValueError('cache_video_count must be positive.')
        if video_names is None:
            video_names = sorted(path.name for path in self.root.glob('*.npz'))
        self.video_names = tuple(str(name) for name in video_names)
        if not self.video_names:
            raise RuntimeError('No videos found in {}'.format(self.root))
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.video_names) * self.views_per_video

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _video(self, video_index):
        video_index = int(video_index)
        cached = self._cache.pop(video_index, None)
        if cached is None:
            cached = load_event_video(self.root / self.video_names[video_index])
        self._cache[video_index] = cached
        while len(self._cache) > self.cache_video_count:
            self._cache.popitem(last=False)
        return cached

    def _choose_start(self, video_index, view_index, video):
        seed = (
            self.random_seed
            + 1000003 * self.current_epoch
            + 1009 * int(video_index)
            + int(view_index)
        )
        rng = np.random.default_rng(seed)
        positive_indices = np.flatnonzero(video['labels'] > 0.5)
        use_positive = (
            positive_indices.size > 0
            and rng.random() < self.positive_clip_probability
        )
        if use_positive:
            anchor = int(positive_indices[rng.integers(positive_indices.size)])
        else:
            if not len(video['locations']):
                raise RuntimeError('video contains no events.')
            anchor = int(rng.integers(len(video['locations'])))
        return _aligned_clip_start(
            video['locations'][anchor, 2],
            self.whole_t,
            self.clip_duration,
            rng,
            self.temporal_bin,
        )

    def __getitem__(self, index):
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError('M88 clip index out of range.')
        video_index, view_index = divmod(index, self.views_per_video)
        video = self._video(video_index)
        start = self._choose_start(video_index, view_index, video)
        clip = build_sparse_clip(
            video,
            start,
            clip_duration=self.clip_duration,
            temporal_bin=self.temporal_bin,
            width=self.width,
            height=self.height,
            count_clip=self.count_clip,
        )
        clip['video_index'] = video_index
        clip['video_name'] = self.video_names[video_index]
        return clip


def single_clip_collate(batch):
    """Keep a variable-size sparse clip unbatched; M88 always uses batch=1."""
    if len(batch) != 1:
        raise ValueError('M88 sparse 3D training requires batch_size=1.')
    return batch[0]
