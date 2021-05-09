import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_features, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_patches, num_channels, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.mlp = MLP(num_patches, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_channels)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_channels, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_channels)
        return x + residual


class ChannelMixer(nn.Module):
    def __init__(self, num_patches, num_channels, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.mlp = MLP(num_channels, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_channels)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_channels)
        return x + residual


class MixerLayer(nn.Module):
    def __init__(self, num_patches, num_channels, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(num_patches, num_channels, dropout)
        self.channel_mixer = ChannelMixer(num_patches, num_channels, dropout)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_channels)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_channels)
        return x


class MLPMixer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=128,
        image_size=224,
        patch_size=16,
        num_layers=8,
        num_classes=10,
        dropout=0.5,
    ):
        sqrt_num_patches, remainder = divmod(image_size, patch_size)
        assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
        num_patches = sqrt_num_patches ** 2
        super().__init__()
        self.patcher = nn.Sequential(
            # per-patch fully-connected is equivalent to strided conv2d
            nn.Conv2d(
                in_channels, out_channels, kernel_size=patch_size, stride=patch_size
            ),
            Rearrange("b c h w -> b (h w) c"),
        )
        self.mixers = nn.Sequential(
            *[MixerLayer(num_patches, out_channels, dropout) for _ in range(num_layers)]
        )
        self.classifier = nn.Linear(out_channels, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        # patches.shape == (batch_size, num_patches, out_channels)
        embedding = self.mixers(patches)
        # out.shape == (batch_size, num_patches, out_channels)
        embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits

