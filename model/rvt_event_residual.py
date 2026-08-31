"""End-to-end RVT-Tiny residual head for per-event segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional

from utils.rvt_component_features import (
    PADDED_HEIGHT,
    PADDED_WIDTH,
    RVT_HEIGHT,
    RVT_WIDTH,
)


def _group_count(channels, maximum=8):
    for groups in range(min(int(maximum), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__()
        padding = int(kernel_size) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class RVTResidualDecoder(nn.Module):
    """Fuse all recurrent stages and recover full-resolution event detail."""

    def __init__(self, stage_dims=(32, 64, 128, 256), fpn_channels=48, detail_channels=24):
        super().__init__()
        self.laterals = nn.ModuleList(
            nn.Conv2d(int(channels), int(fpn_channels), kernel_size=1)
            for channels in stage_dims
        )
        self.fpn_refine = nn.Sequential(
            ConvNormAct(fpn_channels, fpn_channels),
            ConvNormAct(fpn_channels, fpn_channels),
        )
        self.detail = nn.Sequential(
            ConvNormAct(20, detail_channels),
            ConvNormAct(detail_channels, detail_channels),
        )
        self.fuse = nn.Sequential(
            ConvNormAct(fpn_channels + detail_channels, fpn_channels),
            ConvNormAct(fpn_channels, fpn_channels),
        )
        self.output = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, stages, padded_histogram):
        if tuple(sorted(stages)) != (1, 2, 3, 4):
            raise ValueError("RVT decoder requires stages 1 through 4")
        target_size = stages[1].shape[-2:]
        fused = self.laterals[0](stages[1])
        for stage_number, lateral in zip((2, 3, 4), self.laterals[1:]):
            projected = lateral(stages[stage_number])
            fused = fused + functional.interpolate(
                projected,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        fused = self.fpn_refine(fused)
        fused = functional.interpolate(
            fused,
            size=(PADDED_HEIGHT, PADDED_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        normalized_histogram = torch.log1p(padded_histogram) / 4.0
        detail = self.detail(normalized_histogram)
        residual = self.output(self.fuse(torch.cat((fused, detail), dim=1)))
        return residual[:, :, :RVT_HEIGHT, :RVT_WIDTH]


class RVTEventResidualNet(nn.Module):
    """Predict a bounded correction to an existing per-event base logit."""

    def __init__(self, backbone, fpn_channels=48, detail_channels=24, residual_scale=2.0):
        super().__init__()
        self.backbone = backbone
        stage_dims = tuple(int(value) for value in backbone.stage_dims)
        self.decoder = RVTResidualDecoder(
            stage_dims=stage_dims,
            fpn_channels=fpn_channels,
            detail_channels=detail_channels,
        )
        self.residual_scale = float(residual_scale)
        if self.residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")

    def forward_step(self, histogram, states=None):
        if histogram.ndim != 4 or histogram.shape[1:] != (20, RVT_HEIGHT, RVT_WIDTH):
            raise ValueError("histogram must have shape [B,20,240,304]")
        histogram = histogram.float()
        padded = functional.pad(
            histogram,
            (0, PADDED_WIDTH - RVT_WIDTH, 0, PADDED_HEIGHT - RVT_HEIGHT),
        )
        stages, states = self.backbone(padded, prev_states=states)
        raw_residual = self.decoder(stages, padded)
        return self.residual_scale * torch.tanh(raw_residual), states

    def forward(self, histograms, states=None):
        if histograms.ndim != 5 or histograms.shape[2:] != (20, RVT_HEIGHT, RVT_WIDTH):
            raise ValueError("histograms must have shape [B,T,20,240,304]")
        outputs = []
        for step in range(histograms.shape[1]):
            residual, states = self.forward_step(histograms[:, step], states)
            outputs.append(residual)
        return torch.stack(outputs, dim=1), states


def gather_sequence_event_residuals(residual_maps, event_steps, event_y, event_x):
    """Gather residual logits for a batch-size-one recurrent sequence."""
    if residual_maps.ndim != 5 or residual_maps.shape[0] != 1 or residual_maps.shape[2] != 1:
        raise ValueError("residual_maps must have shape [1,T,1,H,W]")
    tensors = (event_steps, event_y, event_x)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("event coordinate tensors must be flat")
    if not (event_steps.shape == event_y.shape == event_x.shape):
        raise ValueError("event coordinate tensors must have matching shapes")
    if event_steps.numel() == 0:
        return residual_maps.reshape(-1)[:0]
    if (
        event_steps.min() < 0
        or event_steps.max() >= residual_maps.shape[1]
        or event_y.min() < 0
        or event_y.max() >= residual_maps.shape[-2]
        or event_x.min() < 0
        or event_x.max() >= residual_maps.shape[-1]
    ):
        raise ValueError("an event coordinate is outside the residual maps")
    return residual_maps[
        0,
        event_steps.long(),
        0,
        event_y.long(),
        event_x.long(),
    ]
