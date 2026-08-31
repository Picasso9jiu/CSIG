"""Persistent object-query head over frozen M26 H/4 decoder features."""

import torch
import torch.nn as nn
import torch.nn.functional as functional


class PersistentTrackQueryNet(nn.Module):
    """Track a fixed query set through one full event video.

    The input is a frozen decoder feature sequence ``[T, C, H, W]``.  Query
    states use spatial cross-attention at each bin and a shared GRU transition,
    then emit existence, normalized centre, scale, and velocity.  It contains
    no event labels, target identifiers, file names, or M26 logits.
    """

    def __init__(self, input_channels, query_count=8, hidden_dim=32, heads=4):
        super().__init__()
        input_channels = int(input_channels)
        query_count = int(query_count)
        hidden_dim = int(hidden_dim)
        heads = int(heads)
        if input_channels <= 0 or query_count <= 0 or hidden_dim <= 0:
            raise ValueError('input_channels, query_count, and hidden_dim must be positive.')
        if hidden_dim % heads != 0:
            raise ValueError('hidden_dim must be divisible by heads.')
        self.input_channels = input_channels
        self.query_count = query_count
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.feature_projection = nn.Conv2d(input_channels, hidden_dim, kernel_size=1)
        self.position_projection = nn.Linear(2, hidden_dim, bias=False)
        self.query_seed = nn.Parameter(torch.empty(query_count, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True,
        )
        self.transition = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.center_head = nn.Linear(hidden_dim, 2)
        self.scale_head = nn.Linear(hidden_dim, 1)
        self.velocity_head = nn.Linear(hidden_dim, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.query_seed, std=0.02)
        nn.init.zeros_(self.existence_head.weight)
        nn.init.constant_(self.existence_head.bias, -2.0)

    @staticmethod
    def _position_grid(height, width, device, dtype):
        y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        # PyTorch 1.9 does not support the ``indexing`` keyword.  Two inputs
        # have always used matrix (ij) indexing, so this is equivalent.
        yy, xx = torch.meshgrid(y, x)
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError('features must have shape [T, C, H, W].')
        if features.shape[1] != self.input_channels:
            raise ValueError(
                'features have {} channels, expected {}.'.format(
                    features.shape[1], self.input_channels
                )
            )
        time_count, _, height, width = features.shape
        if time_count <= 0:
            raise ValueError('features must contain at least one time bin.')
        projected = self.feature_projection(features)
        tokens = projected.flatten(2).transpose(1, 2)
        position = self.position_projection(
            self._position_grid(height, width, features.device, projected.dtype)
        )
        tokens = tokens + position.unsqueeze(0)
        global_features = projected.mean(dim=(2, 3))
        state = self.query_seed
        existence = []
        centers = []
        scales = []
        velocities = []
        for time_index in range(time_count):
            context, _ = self.cross_attention(
                state.unsqueeze(0),
                tokens[time_index:time_index + 1],
                tokens[time_index:time_index + 1],
                need_weights=False,
            )
            global_context = global_features[time_index].expand(self.query_count, -1)
            state = self.transition(
                torch.cat((context.squeeze(0), global_context), dim=1), state
            )
            existence.append(self.existence_head(state).squeeze(1))
            centers.append(torch.sigmoid(self.center_head(state)))
            scales.append(functional.softplus(self.scale_head(state).squeeze(1)) + 1e-4)
            velocities.append(torch.tanh(self.velocity_head(state)))
        return {
            'existence_logits': torch.stack(existence, dim=0),
            'centers': torch.stack(centers, dim=0),
            'scales': torch.stack(scales, dim=0),
            'velocities': torch.stack(velocities, dim=0),
        }
