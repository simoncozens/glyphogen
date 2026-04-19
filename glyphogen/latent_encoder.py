import torch
from torch import nn


class MaskLatentEncoder(nn.Module):
    """Encodes a normalized contour mask into a latent vector."""

    def __init__(
        self,
        image_size: tuple[int, int],
        latent_dim: int = 32,
        rate: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.rate = rate

        target_feature_size = 8
        channel_schedule = [16, 32, 64, 128, 256]
        kernel_schedule = [7, 5, 5, 3, 3]
        in_channels = 1
        current_h, current_w = image_size
        self.num_pyramid_layers = 0

        for i, (out_channels, kernel_size) in enumerate(
            zip(channel_schedule, kernel_schedule), start=1
        ):
            if (
                i > 1
                and current_h <= target_feature_size
                and current_w <= target_feature_size
            ):
                break

            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=2,
            )
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
            norm = nn.LayerNorm([out_channels, current_h, current_w])
            relu = nn.ReLU()

            setattr(self, f"conv{i}", conv)
            setattr(self, f"norm{i}", norm)
            setattr(self, f"relu{i}", relu)

            in_channels = out_channels
            self.num_pyramid_layers += 1

        self.dropout = nn.Dropout(rate)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_channels * current_h * current_w, latent_dim)
        self.norm_dense = nn.LayerNorm(latent_dim)
        self.sigmoid = nn.Sigmoid()
        self.output_dense = nn.Linear(latent_dim, latent_dim)

    @torch.compile
    def forward(self, inputs):
        x = inputs
        for i in range(1, self.num_pyramid_layers + 1):
            x = getattr(self, f"conv{i}")(x)
            x = getattr(self, f"norm{i}")(x)
            x = getattr(self, f"relu{i}")(x)
            x = self.dropout(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.norm_dense(x)
        x = self.sigmoid(x)
        return self.output_dense(x)
