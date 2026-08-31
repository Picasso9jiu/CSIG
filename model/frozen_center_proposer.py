"""Independent centre proposer attached to frozen M26 decoder features."""

import torch.nn as nn


class FrozenCenterProposer(nn.Module):
    """Small H/4 heatmap head with no connection to the event-logit head."""

    def __init__(self, input_channels, hidden_channels=None):
        super().__init__()
        input_channels = int(input_channels)
        hidden_channels = int(hidden_channels or input_channels)
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError('input_channels and hidden_channels must be positive.')
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.projection = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, decoded_features):
        if decoded_features.ndim != 4:
            raise ValueError('decoded_features must have shape [B, C, H, W].')
        if decoded_features.shape[1] != self.input_channels:
            raise ValueError(
                'decoded_features have {} channels, expected {}.'.format(
                    decoded_features.shape[1], self.input_channels
                )
            )
        return self.projection(decoded_features)
