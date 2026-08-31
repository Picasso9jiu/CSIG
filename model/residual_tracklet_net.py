"""Small M75-C motion graph model for independent residual candidate scoring."""

import torch
from torch import nn


class ResidualTrackletNet(nn.Module):
    def __init__(self, scalar_dim, edge_dim=6, hidden_dim=64):
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(32 + scalar_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(2)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.node_head = nn.Linear(hidden_dim, 1)
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, patches, scalar_features, edge_index, edge_features):
        hidden = self.node_encoder(torch.cat((self.patch_encoder(patches), scalar_features), dim=1))
        if edge_index.numel():
            source, destination = edge_index
            for message_layer, norm in zip(self.message_layers, self.norm_layers):
                message_input = torch.cat((hidden[source], hidden[destination], edge_features), dim=1)
                message = message_layer(message_input)
                aggregate = torch.zeros_like(hidden)
                aggregate.index_add_(0, destination, message)
                degree = torch.zeros(hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device)
                degree.index_add_(0, destination, torch.ones_like(message[:, :1]))
                hidden = norm(hidden + aggregate / degree.clamp_min(1.0))
            edge_logits = self.edge_head(torch.cat((hidden[source], hidden[destination], edge_features), dim=1)).reshape(-1)
        else:
            edge_logits = hidden.new_empty((0,))
        return self.node_head(hidden).reshape(-1), edge_logits
