"""Frozen RVT-Tiny features for label-free final-component verification.

The RVT source tree and checkpoint stay external to this repository.  This
module contains only the adapter needed to construct the official Gen1 event
histogram, load the published backbone strictly, and sample recurrent feature
maps at production-component centres.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as functional


# The original M116 adapter targets the Gen1 RVT checkpoint.  M151 adds the
# official 1Mpx (called Gen4 by RVT) checkpoint as a separate, opt-in variant.
# Keeping both hashes here makes it impossible to silently use the wrong
# representation/checkpoint pair.
RVT_TINY_SHA256 = "05f67827744575fac7f840508d360bdeb16d25302e06165c53b6e84b3ef305f9"
RVT_1MPX_TINY_SHA256 = "4298fb5290e3bf6bb0914f37dacb402ed8b759236ad5d211c25056fae7b45ab7"
RVT_1MPX_SMALL_SHA256 = "16d5fc3698fbdbc4598f022a9734c8c01e02d024a273e4237692dfe951a9b7b2"
SOURCE_WIDTH = 346
SOURCE_HEIGHT = 260
RVT_WIDTH = 304
RVT_HEIGHT = 240
PADDED_WIDTH = 320
PADDED_HEIGHT = 256
RVT_1MPX_WIDTH = 640
RVT_1MPX_HEIGHT = 360
RVT_1MPX_PADDED_WIDTH = 640
RVT_1MPX_PADDED_HEIGHT = 384
TIME_BIN_SIZE = 50
MICRO_BINS = 10
FEATURE_DIM = 928
FEATURE_DIM_1MPX_SMALL = 1392


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_external_paths(rvt_repo, dependency_dir):
    for path in (str(Path(dependency_dir).resolve()), str(Path(rvt_repo).resolve())):
        if path not in sys.path:
            sys.path.insert(0, path)


def load_frozen_rvt_tiny(rvt_repo, dependency_dir, checkpoint, device, variant="gen1"):
    """Load a published RVT-Tiny backbone with strict key and variant checks.

    ``variant`` is deliberately explicit.  The 1Mpx checkpoint was trained
    with the Gen4/downsampled input geometry, whose stage-1 attention partition
    is ``[6, 10]`` for the padded ``384x640`` canvas.  Its tensor keys are
    otherwise identical to the Gen1 Tiny model, so a checksum plus the proper
    partition is the important compatibility guard.
    """
    if variant not in ("gen1", "1mpx", "1mpx_small"):
        raise ValueError("variant must be 'gen1', '1mpx', or '1mpx_small'")
    if variant == "gen1":
        expected_checksum = RVT_TINY_SHA256
        partition_size = [8, 10]
        embed_dim = 32
        dim_head = 32
    elif variant == "1mpx":
        expected_checksum = RVT_1MPX_TINY_SHA256
        partition_size = [6, 10]
        embed_dim = 32
        dim_head = 32
    else:
        expected_checksum = RVT_1MPX_SMALL_SHA256
        partition_size = [6, 10]
        embed_dim = 48
        dim_head = 24
    _add_external_paths(rvt_repo, dependency_dir)
    from omegaconf import OmegaConf
    from models.detection.recurrent_backbone.maxvit_rnn import RNNDetector

    checkpoint = Path(checkpoint)
    checksum = _sha256(checkpoint)
    if checksum != expected_checksum:
        raise RuntimeError(
            "Unexpected RVT-Tiny checkpoint SHA256: {} (expected {})".format(
                checksum, expected_checksum
            )
        )
    config = OmegaConf.create(
        {
            "input_channels": 20,
            "embed_dim": embed_dim,
            "dim_multiplier": [1, 2, 4, 8],
            "num_blocks": [1, 1, 1, 1],
            "T_max_chrono_init": [4, 8, 16, 32],
            "enable_masking": False,
            "stem": {"patch_size": 4},
            "stage": {
                "downsample": {
                    "type": "patch",
                    "overlap": True,
                    "norm_affine": True,
                },
                "attention": {
                    "use_torch_mha": False,
                    "partition_size": partition_size,
                    "dim_head": dim_head,
                    "attention_bias": True,
                    "mlp_activation": "gelu",
                    "mlp_gated": False,
                    "mlp_bias": True,
                    "mlp_ratio": 4,
                    "drop_mlp": 0,
                    "drop_path": 0,
                    "ls_init_value": 1e-5,
                },
                "lstm": {
                    "dws_conv": False,
                    "dws_conv_only_hidden": True,
                    "dws_conv_kernel_size": 3,
                    "drop_cell_update": 0,
                },
            },
        }
    )
    model = RNNDetector(config)
    payload = torch.load(str(checkpoint), map_location="cpu")
    published = payload.get("state_dict")
    if not isinstance(published, dict):
        raise ValueError("RVT checkpoint has no Lightning state_dict.")
    prefix = "mdl.backbone."
    state = {
        key[len(prefix) :]: value
        for key, value in published.items()
        if key.startswith(prefix)
    }
    if len(state) != len(model.state_dict()):
        raise RuntimeError(
            "RVT backbone tensor count mismatch: {} vs {}".format(
                len(state), len(model.state_dict())
            )
        )
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "variant": variant,
        "checkpoint_sha256": checksum,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "parameter_count": int(sum(item.numel() for item in model.parameters())),
        "stage_dims": list(model.stage_dims),
        "stage_strides": list(model.strides),
        "feature_dim": FEATURE_DIM if variant != "1mpx_small" else FEATURE_DIM_1MPX_SMALL,
    }


def _scale_event_coordinates(locations, width, height):
    locations = np.asarray(locations)
    x = np.floor(locations[:, 0].astype(np.float64) * width / SOURCE_WIDTH)
    y = np.floor(locations[:, 1].astype(np.float64) * height / SOURCE_HEIGHT)
    x = np.clip(x.astype(np.int64), 0, width - 1)
    y = np.clip(y.astype(np.int64), 0, height - 1)
    return x, y


def scale_event_coordinates(locations):
    """Map EV-UAV coordinates into RVT Gen1's 304x240 canvas."""
    return _scale_event_coordinates(locations, RVT_WIDTH, RVT_HEIGHT)


def scale_event_coordinates_1mpx(locations):
    """Map EV-UAV coordinates into the RVT 1Mpx downsampled canvas."""
    return _scale_event_coordinates(locations, RVT_1MPX_WIDTH, RVT_1MPX_HEIGHT)


def _build_rvt_histogram(locations, polarities, time_bin, width, height):
    locations = np.asarray(locations)
    polarities = np.asarray(polarities, dtype=np.int64).reshape(-1)
    timestamps = locations[:, 2].astype(np.int64, copy=False)
    start = int(time_bin) * TIME_BIN_SIZE
    stop = start + TIME_BIN_SIZE
    left = int(np.searchsorted(timestamps, start, side="left"))
    right = int(np.searchsorted(timestamps, stop, side="left"))
    histogram = np.zeros((2, MICRO_BINS, height, width), dtype=np.uint8)
    if right <= left:
        return histogram.reshape(2 * MICRO_BINS, height, width)
    local = locations[left:right]
    local_polarity = polarities[left:right]
    if np.any((local_polarity < 0) | (local_polarity > 1)):
        raise ValueError("RVT input polarity must be in {0,1}.")
    local_t = local[:, 2].astype(np.int64, copy=False)
    span = max(int(local_t[-1] - local_t[0]), 1)
    micro = np.floor((local_t - local_t[0]).astype(np.float64) / span * MICRO_BINS)
    micro = np.clip(micro.astype(np.int64), 0, MICRO_BINS - 1)
    x, y = _scale_event_coordinates(local, width, height)
    # Use int16 accumulation and clip to reproduce RVT's safe uint8 contract.
    safe = np.zeros_like(histogram, dtype=np.int16)
    np.add.at(safe, (local_polarity, micro, y, x), 1)
    np.clip(safe, 0, 255, out=safe)
    return safe.astype(np.uint8).reshape(2 * MICRO_BINS, height, width)


def build_rvt_histogram(locations, polarities, time_bin):
    """Build RVT Gen1's uint8 10-bin x 2-polarity representation."""
    output = _build_rvt_histogram(locations, polarities, time_bin, RVT_WIDTH, RVT_HEIGHT)
    return output.reshape(2 * MICRO_BINS, RVT_HEIGHT, RVT_WIDTH)


def build_rvt_histogram_1mpx(locations, polarities, time_bin):
    """Build the RVT 1Mpx/Gen4 downsampled representation."""
    output = _build_rvt_histogram(
        locations, polarities, time_bin, RVT_1MPX_WIDTH, RVT_1MPX_HEIGHT
    )
    return output.reshape(2 * MICRO_BINS, RVT_1MPX_HEIGHT, RVT_1MPX_WIDTH)


def _scale_component_centres(centres, width, height):
    centres = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    output = np.empty_like(centres)
    output[:, 0] = centres[:, 0] * width / SOURCE_WIDTH
    output[:, 1] = centres[:, 1] * height / SOURCE_HEIGHT
    return output


def scale_component_centres(centres):
    return _scale_component_centres(centres, RVT_WIDTH, RVT_HEIGHT)


def scale_component_centres_1mpx(centres):
    return _scale_component_centres(centres, RVT_1MPX_WIDTH, RVT_1MPX_HEIGHT)


def _sample_stage(stage, centres, stride, include_pooling):
    if len(centres) == 0:
        multiplier = 3 if include_pooling else 1
        return stage.new_empty((0, stage.shape[1] * multiplier))
    x = torch.from_numpy(np.floor(centres[:, 0] / stride).astype(np.int64)).to(
        device=stage.device
    )
    y = torch.from_numpy(np.floor(centres[:, 1] / stride).astype(np.int64)).to(
        device=stage.device
    )
    x.clamp_(0, stage.shape[-1] - 1)
    y.clamp_(0, stage.shape[-2] - 1)
    centre = stage[0, :, y, x].transpose(0, 1)
    if not include_pooling:
        return centre
    average = functional.avg_pool2d(stage, kernel_size=3, stride=1, padding=1)
    maximum = functional.max_pool2d(stage, kernel_size=3, stride=1, padding=1)
    return torch.cat(
        (
            centre,
            average[0, :, y, x].transpose(0, 1),
            maximum[0, :, y, x].transpose(0, 1),
        ),
        dim=1,
    )


def extract_component_features(
    model,
    locations,
    polarities,
    components_by_bin,
    device,
    whole_time=8000,
    use_amp=True,
    variant="gen1",
):
    """Run a complete recurrent video and return one feature per component."""
    if variant not in ("gen1", "1mpx", "1mpx_small"):
        raise ValueError("variant must be 'gen1', '1mpx', or '1mpx_small'")
    is_gen1 = variant == "gen1"
    width = RVT_WIDTH if is_gen1 else RVT_1MPX_WIDTH
    height = RVT_HEIGHT if is_gen1 else RVT_1MPX_HEIGHT
    padded_width = PADDED_WIDTH if is_gen1 else RVT_1MPX_PADDED_WIDTH
    padded_height = PADDED_HEIGHT if is_gen1 else RVT_1MPX_PADDED_HEIGHT
    histogram_builder = build_rvt_histogram if is_gen1 else build_rvt_histogram_1mpx
    centre_scaler = scale_component_centres if is_gen1 else scale_component_centres_1mpx
    expected_feature_dim = FEATURE_DIM if variant != "1mpx_small" else FEATURE_DIM_1MPX_SMALL
    locations = np.asarray(locations, dtype=np.int64)
    polarities = np.asarray(polarities, dtype=np.int64)
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("locations must have shape [N,3].")
    if len(locations) != len(polarities):
        raise ValueError("location/polarity event counts differ.")
    if len(locations) and np.any(np.diff(locations[:, 2]) < 0):
        raise ValueError("RVT feature extraction requires time-sorted events.")
    time_bins = int(whole_time) // TIME_BIN_SIZE
    if time_bins * TIME_BIN_SIZE != int(whole_time):
        raise ValueError("whole_time must be divisible by 50.")
    features = {}
    states = None
    with torch.no_grad():
        for time_bin in range(time_bins):
            histogram = histogram_builder(locations, polarities, time_bin)
            tensor = torch.from_numpy(histogram).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            tensor = functional.pad(
                tensor,
                (0, padded_width - width, 0, padded_height - height),
            )
            with torch.cuda.amp.autocast(enabled=bool(use_amp)):
                stages, states = model(tensor, prev_states=states)
            components = components_by_bin.get(time_bin, ())
            if not components:
                continue
            centres = centre_scaler(
                np.stack([item["centroid"] for item in components])
            )
            sampled = torch.cat(
                (
                    _sample_stage(stages[1], centres, 4, True),
                    _sample_stage(stages[2], centres, 8, True),
                    _sample_stage(stages[3], centres, 16, True),
                    _sample_stage(stages[4], centres, 32, False),
                ),
                dim=1,
            )
            if sampled.shape[1] != expected_feature_dim:
                raise AssertionError(
                    "Unexpected RVT component feature width {}".format(sampled.shape[1])
                )
            for component, vector in zip(components, sampled.float().cpu()):
                features[int(component["component_index"])] = vector
    expected = sum(len(items) for items in components_by_bin.values())
    if len(features) != expected:
        raise RuntimeError("RVT did not produce every component feature.")
    return torch.stack([features[index] for index in range(expected)]).contiguous()
