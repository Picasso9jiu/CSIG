"""Metric-time sequences for end-to-end RVT residual segmentation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.rvt_component_features import (
    MICRO_BINS,
    RVT_HEIGHT,
    RVT_WIDTH,
    TIME_BIN_SIZE,
    build_rvt_histogram,
    scale_event_coordinates,
)


def load_score_records(path):
    payload = torch.load(str(path), map_location="cpu")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("score cache has no records list")
    return records


def load_fold_map(path):
    import json

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    by_name = {str(row["name"]): int(row["fold"]) for row in manifest["records"]}
    return manifest, by_name


def _record_name(record):
    return Path(str(record["name"])).name


def _record_arrays(record, raw_path):
    with np.load(str(raw_path)) as payload:
        raw_locations = np.asarray(payload["ev_loc"], dtype=np.int64)
        raw_features = np.asarray(payload["evs_norm"])
    cached_locations = torch.as_tensor(record["locs"]).cpu().numpy()[:, 1:4]
    if not np.array_equal(raw_locations, cached_locations):
        raise AssertionError("raw NPZ and score cache locations do not align")
    if raw_features.shape[0] != raw_locations.shape[0] or raw_features.shape[1] < 4:
        raise ValueError("raw NPZ has an invalid evs_norm array")
    polarities = np.rint(raw_features[:, 3]).astype(np.int64)
    if np.any((polarities < 0) | (polarities > 1)):
        raise ValueError("raw polarity must be in {0,1}")
    labels = torch.as_tensor(record["seg_label"]).cpu().float().numpy() > 0.5
    target_ids = np.asarray(record["idx_label"], dtype=np.int64)
    base_scores = torch.as_tensor(record["scores"]).cpu().float().numpy().reshape(-1)
    event_count = len(raw_locations)
    if not (
        len(polarities)
        == len(labels)
        == len(target_ids)
        == len(base_scores)
        == event_count
    ):
        raise AssertionError("raw and cached event arrays have different lengths")
    if not np.isfinite(base_scores).all():
        raise ValueError("base score cache contains non-finite values")
    if np.any(np.diff(raw_locations[:, 2]) < 0):
        raise ValueError("events must be sorted by timestamp")
    return raw_locations, polarities, labels, target_ids, base_scores


class RVTResidualSequenceDataset(Dataset):
    """Deterministic random contiguous sequences from selected video folds."""

    def __init__(
        self,
        score_cache,
        data_root,
        fold_manifest,
        include_folds,
        sequence_length=8,
        views_per_video=2,
        positive_sequence_probability=0.75,
        random_seed=121,
        dense_event_count_cutoff=200000,
        dense_view_multiplier=2,
        whole_time=8000,
        cache_videos=True,
    ):
        super().__init__()
        self.records = load_score_records(score_cache)
        self.data_root = Path(data_root)
        self.manifest, fold_by_name = load_fold_map(fold_manifest)
        self.include_folds = tuple(sorted({int(value) for value in include_folds}))
        self.sequence_length = int(sequence_length)
        self.views_per_video = int(views_per_video)
        self.positive_sequence_probability = float(positive_sequence_probability)
        self.random_seed = int(random_seed)
        self.dense_event_count_cutoff = int(dense_event_count_cutoff)
        self.dense_view_multiplier = int(dense_view_multiplier)
        self.whole_time = int(whole_time)
        self.time_bin_count = self.whole_time // TIME_BIN_SIZE
        self.cache_videos = bool(cache_videos)
        self.current_epoch = 0
        if self.sequence_length <= 0 or self.sequence_length > self.time_bin_count:
            raise ValueError("sequence_length is outside the video time range")
        if self.views_per_video <= 0 or self.dense_view_multiplier <= 0:
            raise ValueError("view counts must be positive")
        if not 0.0 <= self.positive_sequence_probability <= 1.0:
            raise ValueError("positive_sequence_probability must be in [0,1]")
        if self.whole_time <= 0 or self.whole_time % TIME_BIN_SIZE:
            raise ValueError("whole_time must be a positive multiple of 50")
        if not self.include_folds:
            raise ValueError("include_folds must not be empty")

        self.record_indices = []
        for index, record in enumerate(self.records):
            name = _record_name(record)
            if name not in fold_by_name:
                raise KeyError("fold manifest is missing {}".format(name))
            if int(fold_by_name[name]) in self.include_folds:
                self.record_indices.append(index)
        if not self.record_indices:
            raise ValueError("selected folds contain no score records")

        self.schedule = []
        for record_index in self.record_indices:
            event_count = int(self.records[record_index]["event_count"])
            multiplier = (
                self.dense_view_multiplier
                if event_count >= self.dense_event_count_cutoff
                else 1
            )
            for view_index in range(self.views_per_video * multiplier):
                self.schedule.append((record_index, view_index))
        self._video_cache = {}
        if self.cache_videos:
            for record_index in self.record_indices:
                self._video(record_index)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def __len__(self):
        return len(self.schedule)

    def _video(self, record_index):
        cached = self._video_cache.get(int(record_index))
        if cached is not None:
            return cached
        record = self.records[int(record_index)]
        name = _record_name(record)
        arrays = _record_arrays(record, self.data_root / name)
        locations, polarities, labels, target_ids, base_scores = arrays
        bins = np.clip(
            locations[:, 2] // TIME_BIN_SIZE,
            0,
            self.time_bin_count - 1,
        ).astype(np.int64, copy=False)
        indices_by_bin = tuple(
            np.flatnonzero(bins == time_bin).astype(np.int64, copy=False)
            for time_bin in range(self.time_bin_count)
        )
        occupied_bins = np.flatnonzero(
            np.bincount(bins, minlength=self.time_bin_count) > 0
        ).astype(np.int64, copy=False)
        positive_bins = np.flatnonzero(
            np.bincount(bins[labels], minlength=self.time_bin_count) > 0
        ).astype(np.int64, copy=False)
        scaled_x, scaled_y = scale_event_coordinates(locations)
        cached = {
            "name": name,
            "locations": locations,
            "polarities": polarities,
            "labels": labels,
            "target_ids": target_ids,
            "base_scores": base_scores,
            "indices_by_bin": indices_by_bin,
            "occupied_bins": occupied_bins,
            "positive_bins": positive_bins,
            "scaled_x": scaled_x,
            "scaled_y": scaled_y,
        }
        if self.cache_videos:
            self._video_cache[int(record_index)] = cached
        return cached

    def _sequence_start(self, record_index, view_index, video):
        generator = np.random.default_rng(
            self.random_seed
            + self.current_epoch * 1000003
            + int(record_index) * 1009
            + int(view_index)
        )
        use_positive = bool(
            video["positive_bins"].size
            and generator.random() < self.positive_sequence_probability
        )
        candidates = (
            video["positive_bins"] if use_positive else video["occupied_bins"]
        )
        if candidates.size == 0:
            raise RuntimeError("{} contains no occupied time bin".format(video["name"]))
        anchor = int(candidates[generator.integers(candidates.size)])
        anchor_offset = int(generator.integers(self.sequence_length))
        return int(
            np.clip(
                anchor - anchor_offset,
                0,
                self.time_bin_count - self.sequence_length,
            )
        )

    def __getitem__(self, index):
        record_index, view_index = self.schedule[int(index)]
        video = self._video(record_index)
        start_bin = self._sequence_start(record_index, view_index, video)
        histograms = []
        event_indices = []
        event_steps = []
        for step in range(self.sequence_length):
            time_bin = start_bin + step
            histogram = build_rvt_histogram(
                video["locations"], video["polarities"], time_bin
            )
            if histogram.shape != (2 * MICRO_BINS, RVT_HEIGHT, RVT_WIDTH):
                raise AssertionError("RVT histogram has an unexpected shape")
            histograms.append(histogram)
            members = video["indices_by_bin"][time_bin]
            if members.size:
                event_indices.append(members)
                event_steps.append(np.full(members.size, step, dtype=np.int64))
        if not event_indices:
            raise RuntimeError("sampled RVT sequence contains no events")
        event_indices = np.concatenate(event_indices)
        event_steps = np.concatenate(event_steps)
        return {
            "histograms": torch.from_numpy(np.stack(histograms)).to(torch.uint8),
            "event_steps": torch.from_numpy(event_steps),
            "event_x": torch.from_numpy(video["scaled_x"][event_indices]),
            "event_y": torch.from_numpy(video["scaled_y"][event_indices]),
            "labels": torch.from_numpy(
                video["labels"][event_indices].astype(np.float32, copy=False)
            ),
            "target_ids": torch.from_numpy(
                video["target_ids"][event_indices].astype(np.int64, copy=False)
            ),
            "base_scores": torch.from_numpy(
                video["base_scores"][event_indices].astype(np.float32, copy=False)
            ),
            "event_indices": torch.from_numpy(event_indices),
            "record_index": torch.tensor(record_index, dtype=torch.long),
            "start_bin": torch.tensor(start_bin, dtype=torch.long),
        }


def rvt_residual_collate(samples):
    """The first M121 probe intentionally uses one recurrent video at a time."""
    if len(samples) != 1:
        raise ValueError("M121 currently requires batch_size=1")
    return samples[0]
