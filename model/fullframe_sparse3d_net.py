"""Compact full-frame sparse 3D network used by the M88 screen.

This model is intentionally independent of the 50-unit frame/ConvGRU M26
pipeline.  It consumes raw full-frame sparse voxels from a short continuous
time clip and returns one logit for every active voxel.
"""

from __future__ import annotations

import torch
from torch import nn

import spconv.pytorch as spconv


def _norm_relu(channels):
    return nn.Sequential(
        nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class SparseResidualBlock(spconv.SparseModule):
    """Two submanifold convolutions without changing the active coordinates."""

    def __init__(self, channels, indice_key):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
        )
        self.norm1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.conv2 = spconv.SubMConv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
            indice_key=indice_key,
        )
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, value):
        identity = value.features
        value = self.conv1(value)
        value = value.replace_feature(self.activation(self.norm1(value.features)))
        value = self.conv2(value)
        value = value.replace_feature(self.norm2(value.features) + identity)
        return value.replace_feature(self.activation(value.features))


class FullFrameSparse3DNet(nn.Module):
    """Small sparse U-Net for complete 346x260x40 event clips.

    The three strided stages give each event a substantially larger receptive
    field than M83's local patch while keeping the input clip small enough for
    the 4 GB GPU.  Skip connections retain the native-pixel output support.
    """

    def __init__(self, input_channels=3, width=8):
        super().__init__()
        width = int(width)
        if width <= 0:
            raise ValueError('width must be positive.')
        channels = (width, width * 2, width * 3, width * 4)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(
                int(input_channels),
                channels[0],
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='m85_subm0',
            ),
            nn.BatchNorm1d(channels[0], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.enc0 = SparseResidualBlock(channels[0], 'm85_subm0')

        self.down1 = spconv.SparseSequential(
            spconv.SparseConv3d(
                channels[0],
                channels[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
                indice_key='m85_down1',
            ),
            nn.BatchNorm1d(channels[1], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.enc1 = SparseResidualBlock(channels[1], 'm85_subm1')

        self.down2 = spconv.SparseSequential(
            spconv.SparseConv3d(
                channels[1],
                channels[2],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
                indice_key='m85_down2',
            ),
            nn.BatchNorm1d(channels[2], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.enc2 = SparseResidualBlock(channels[2], 'm85_subm2')

        self.down3 = spconv.SparseSequential(
            spconv.SparseConv3d(
                channels[2],
                channels[3],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
                indice_key='m85_down3',
            ),
            nn.BatchNorm1d(channels[3], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = SparseResidualBlock(channels[3], 'm85_subm3')

        self.up2 = spconv.SparseInverseConv3d(
            channels[3],
            channels[2],
            kernel_size=2,
            indice_key='m85_down3',
            bias=False,
        )
        self.up2_norm = _norm_relu(channels[2])
        self.dec2 = spconv.SparseSequential(
            spconv.SubMConv3d(
                channels[2] * 2,
                channels[2],
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='m85_subm2',
            ),
            nn.BatchNorm1d(channels[2], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            SparseResidualBlock(channels[2], 'm85_subm2'),
        )

        self.up1 = spconv.SparseInverseConv3d(
            channels[2],
            channels[1],
            kernel_size=2,
            indice_key='m85_down2',
            bias=False,
        )
        self.up1_norm = _norm_relu(channels[1])
        self.dec1 = spconv.SparseSequential(
            spconv.SubMConv3d(
                channels[1] * 2,
                channels[1],
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='m85_subm1',
            ),
            nn.BatchNorm1d(channels[1], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            SparseResidualBlock(channels[1], 'm85_subm1'),
        )

        self.up0 = spconv.SparseInverseConv3d(
            channels[1],
            channels[0],
            kernel_size=2,
            indice_key='m85_down1',
            bias=False,
        )
        self.up0_norm = _norm_relu(channels[0])
        self.dec0 = spconv.SparseSequential(
            spconv.SubMConv3d(
                channels[0] * 2,
                channels[0],
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='m85_subm0',
            ),
            nn.BatchNorm1d(channels[0], eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            SparseResidualBlock(channels[0], 'm85_subm0'),
        )
        self.head = nn.Linear(channels[0], 1)

    @staticmethod
    def _merge(skip, upsampled):
        if not torch.equal(skip.indices, upsampled.indices):
            raise RuntimeError(
                'Sparse inverse convolution did not restore skip coordinates.'
            )
        return upsampled.replace_feature(
            torch.cat((upsampled.features, skip.features), dim=1)
        )

    def forward(self, sparse_input):
        level0 = self.enc0(self.stem(sparse_input))
        level1 = self.enc1(self.down1(level0))
        level2 = self.enc2(self.down2(level1))
        level3 = self.bottleneck(self.down3(level2))

        up2 = self.up2(level3)
        up2 = up2.replace_feature(self.up2_norm(up2.features))
        level2 = self.dec2(self._merge(level2, up2))

        up1 = self.up1(level2)
        up1 = up1.replace_feature(self.up1_norm(up1.features))
        level1 = self.dec1(self._merge(level1, up1))

        up0 = self.up0(level1)
        up0 = up0.replace_feature(self.up0_norm(up0.features))
        level0 = self.dec0(self._merge(level0, up0))
        return self.head(level0.features).reshape(-1)
