"""Neural encoders for conventional and bilateral respiratory representations."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels, output_channels, kernel_size=7, stride=stride, padding=3, bias=False
            ),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(groups, output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(values) + self.skip(values))


class TemporalEncoder(nn.Module):
    def __init__(self, input_channels: int, output_dimension: int, base_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=11, stride=2, padding=5, bias=False),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.SiLU(),
            ResidualBlock(base_channels, base_channels),
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 4, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        dilation: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = 2 * dilation
        groups = min(8, output_channels)
        self.main = nn.Sequential(
            nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size=5,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv1d(
                output_channels,
                output_channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(groups, output_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(self.main(values) + self.skip(values))


class MultiScaleTemporalEncoder(nn.Module):
    """Parallel short/breath-cycle/long kernels followed by a dilated hierarchy."""

    def __init__(self, input_channels: int, output_dimension: int, base_channels: int) -> None:
        super().__init__()
        self.stem_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels,
                    base_channels,
                    kernel_size=kernel,
                    stride=2,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in (7, 21, 41)
            ]
        )
        self.stem_fusion = nn.Sequential(
            nn.Conv1d(base_channels * 3, base_channels * 2, 1, bias=False),
            nn.GroupNorm(min(8, base_channels * 2), base_channels * 2),
            nn.SiLU(),
        )
        self.network = nn.Sequential(
            DilatedResidualBlock(base_channels * 2, base_channels * 2, dilation=1),
            DilatedResidualBlock(
                base_channels * 2, base_channels * 4, dilation=2, stride=2
            ),
            DilatedResidualBlock(base_channels * 4, base_channels * 4, dilation=4),
            DilatedResidualBlock(
                base_channels * 4, base_channels * 8, dilation=8, stride=2
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stem = torch.cat([branch(values) for branch in self.stem_branches], dim=1)
        return self.network(self.stem_fusion(stem))


class SpectralEncoder(nn.Module):
    """Log-magnitude bilateral spectrum for 180-sample windows."""

    def __init__(self, input_channels: int, output_dimension: int, window_samples: int = 180) -> None:
        super().__init__()
        bins_without_dc = window_samples // 2
        hidden = max(64, output_dimension * 2)
        self.network = nn.Sequential(
            nn.Linear(input_channels * bins_without_dc, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, output_dimension),
            nn.LayerNorm(output_dimension),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(values.float(), dim=-1).abs()[..., 1:]
        magnitude = torch.log1p(spectrum)
        magnitude = (magnitude - magnitude.mean(dim=-1, keepdim=True)) / magnitude.std(
            dim=-1, keepdim=True
        ).clamp_min(1e-5)
        return self.network(magnitude.flatten(1))


class TwoChannelEncoder(nn.Module):
    """General learned baseline that treats L/R as ordinary input channels."""

    def __init__(self, embedding_dimension: int = 256, base_channels: int = 32) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(2, embedding_dimension, base_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(values), dim=1)


def bilateral_channels(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stacked V3 channels and per-nostril availability indicators."""

    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"expected (batch, 2, samples), found {tuple(values.shape)}")
    availability = values.abs().amax(dim=-1).gt(1e-8).to(values.dtype)
    left = values[:, 0:1]
    right = values[:, 1:2]
    left_mask = availability[:, 0:1, None].expand_as(left)
    right_mask = availability[:, 1:2, None].expand_as(right)
    stacked = torch.cat(
        (left, right, left + right, left - right, left_mask, right_mask), dim=1
    )
    return stacked, availability


class V3StackedChannelEncoder(nn.Module):
    """V3 parameter-matched baseline over L, R, sum, difference, and masks."""

    def __init__(self, embedding_dimension: int = 256, base_channels: int = 72) -> None:
        super().__init__()
        self.encoder = TemporalEncoder(6, embedding_dimension, base_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stacked, _ = bilateral_channels(values)
        return F.normalize(self.encoder(stacked), dim=1)


class V3SharedTwoTowerEncoder(nn.Module):
    """Shared per-channel encoder without explicit bilateral interaction paths.

    This comparator isolates weight sharing and late fusion from BIE's
    sum/difference temporal branches and explicit absolute-difference/product
    interactions.  A wider shared tower can be used to match BIE's parameter
    budget without reintroducing those interaction features.
    """

    def __init__(
        self,
        embedding_dimension: int = 256,
        branch_dimension: int = 128,
        base_channels: int = 72,
    ) -> None:
        super().__init__()
        self.channel_encoder = TemporalEncoder(
            2, branch_dimension, base_channels
        )
        self.projection = nn.Sequential(
            nn.Linear(branch_dimension * 2 + 2, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stacked, availability = bilateral_channels(values)
        left_input = torch.cat((stacked[:, 0:1], stacked[:, 4:5]), dim=1)
        right_input = torch.cat((stacked[:, 1:2], stacked[:, 5:6]), dim=1)
        h_left = self.channel_encoder(left_input)
        h_right = self.channel_encoder(right_input)
        return F.normalize(
            self.projection(torch.cat((h_left, h_right, availability), dim=1)),
            dim=1,
        )


class V3UnilateralEncoder(nn.Module):
    """Capacity-matched encoder restricted to exactly one nasal channel."""

    def __init__(
        self,
        embedding_dimension: int = 256,
        base_channels: int = 72,
        *,
        channel_index: int,
    ) -> None:
        super().__init__()
        if channel_index not in (0, 1):
            raise ValueError("channel_index must be zero or one")
        self.channel_index = channel_index
        self.encoder = TemporalEncoder(2, embedding_dimension, base_channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(f"expected (batch, 2, samples), found {tuple(values.shape)}")
        signal = values[:, self.channel_index : self.channel_index + 1]
        availability = signal.abs().amax(dim=-1, keepdim=True).gt(1e-8).to(values.dtype)
        mask = availability.expand_as(signal)
        return F.normalize(self.encoder(torch.cat((signal, mask), dim=1)), dim=1)


class V3BilateralInteractionEncoder(nn.Module):
    """Capacity-matched V3 BIE with shared nostril, global, and asymmetry paths."""

    def __init__(
        self,
        embedding_dimension: int = 256,
        branch_dimension: int = 128,
        base_channels: int = 40,
    ) -> None:
        super().__init__()
        self.nostril_encoder = TemporalEncoder(2, branch_dimension, base_channels)
        self.global_encoder = TemporalEncoder(1, branch_dimension, base_channels)
        self.asymmetry_encoder = TemporalEncoder(1, branch_dimension, base_channels)
        self.projection = nn.Sequential(
            nn.Linear(branch_dimension * 6 + 2, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stacked, availability = bilateral_channels(values)
        left = stacked[:, 0:1]
        right = stacked[:, 1:2]
        left_mask = stacked[:, 4:5]
        right_mask = stacked[:, 5:6]
        h_left = self.nostril_encoder(torch.cat((left, left_mask), dim=1))
        h_right = self.nostril_encoder(torch.cat((right, right_mask), dim=1))
        h_global = self.global_encoder(stacked[:, 2:3])
        h_asymmetry = self.asymmetry_encoder(stacked[:, 3:4])
        combined = torch.cat(
            (
                h_left,
                h_right,
                torch.abs(h_left - h_right),
                h_left * h_right,
                h_global,
                h_asymmetry,
                availability,
            ),
            dim=1,
        )
        return F.normalize(self.projection(combined), dim=1)


class V3BIEAblationEncoder(nn.Module):
    """Prospective V3 BIE component ablations with an unchanged output size."""

    VARIANTS = (
        "no_asymmetry",
        "no_interaction",
        "no_global",
        "independent_nostrils",
    )

    def __init__(
        self,
        variant: str,
        embedding_dimension: int = 256,
        branch_dimension: int = 128,
        base_channels: int = 40,
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown BIE ablation {variant!r}")
        self.variant = variant
        if variant == "independent_nostrils":
            self.left_nostril_encoder = TemporalEncoder(
                2, branch_dimension, base_channels
            )
            self.right_nostril_encoder = TemporalEncoder(
                2, branch_dimension, base_channels
            )
            self.nostril_encoder = None
        else:
            self.nostril_encoder = TemporalEncoder(2, branch_dimension, base_channels)
            self.left_nostril_encoder = None
            self.right_nostril_encoder = None
        self.global_encoder = (
            None
            if variant == "no_global"
            else TemporalEncoder(1, branch_dimension, base_channels)
        )
        self.asymmetry_encoder = (
            None
            if variant == "no_asymmetry"
            else TemporalEncoder(1, branch_dimension, base_channels)
        )
        branch_count = 2
        if variant != "no_interaction":
            branch_count += 2
        if self.global_encoder is not None:
            branch_count += 1
        if self.asymmetry_encoder is not None:
            branch_count += 1
        self.projection = nn.Sequential(
            nn.Linear(branch_dimension * branch_count + 2, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        stacked, availability = bilateral_channels(values)
        left_input = torch.cat((stacked[:, 0:1], stacked[:, 4:5]), dim=1)
        right_input = torch.cat((stacked[:, 1:2], stacked[:, 5:6]), dim=1)
        if self.variant == "independent_nostrils":
            h_left = self.left_nostril_encoder(left_input)
            h_right = self.right_nostril_encoder(right_input)
        else:
            h_left = self.nostril_encoder(left_input)
            h_right = self.nostril_encoder(right_input)
        branches = [h_left, h_right]
        if self.variant != "no_interaction":
            branches.extend((torch.abs(h_left - h_right), h_left * h_right))
        if self.global_encoder is not None:
            branches.append(self.global_encoder(stacked[:, 2:3]))
        if self.asymmetry_encoder is not None:
            branches.append(self.asymmetry_encoder(stacked[:, 3:4]))
        branches.append(availability)
        return F.normalize(self.projection(torch.cat(branches, dim=1)), dim=1)


class MultiScaleTwoChannelEncoder(nn.Module):
    """Stronger general baseline combining multiscale temporal and spectral cues."""

    def __init__(self, embedding_dimension: int = 256, base_channels: int = 24) -> None:
        super().__init__()
        spectral_dimension = max(32, embedding_dimension // 2)
        self.temporal = MultiScaleTemporalEncoder(2, embedding_dimension, base_channels)
        self.spectral = SpectralEncoder(2, spectral_dimension)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dimension + spectral_dimension, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((self.temporal(values), self.spectral(values)), dim=1)
        return F.normalize(self.projection(combined), dim=1)


class BilateralInteractionEncoder(nn.Module):
    """Shared nostril encoding plus global, asymmetry, and interaction branches."""

    def __init__(
        self,
        embedding_dimension: int = 256,
        branch_dimension: int = 64,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.nostril_encoder = TemporalEncoder(1, branch_dimension, base_channels)
        self.global_encoder = TemporalEncoder(1, branch_dimension, base_channels)
        self.asymmetry_encoder = TemporalEncoder(1, branch_dimension, base_channels)
        self.projection = nn.Sequential(
            nn.Linear(branch_dimension * 6, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(f"expected (batch, 2, samples), found {tuple(values.shape)}")
        left = values[:, 0:1]
        right = values[:, 1:2]
        h_left = self.nostril_encoder(left)
        h_right = self.nostril_encoder(right)
        h_global = self.global_encoder(left + right)
        h_asymmetry = self.asymmetry_encoder(left - right)
        combined = torch.cat(
            (
                h_left,
                h_right,
                torch.abs(h_left - h_right),
                h_left * h_right,
                h_global,
                h_asymmetry,
            ),
            dim=1,
        )
        return F.normalize(self.projection(combined), dim=1)


class BilateralAttentionEncoder(nn.Module):
    """Multiscale bilateral tokens fused by self-attention and a learned summary token."""

    def __init__(
        self,
        embedding_dimension: int = 256,
        branch_dimension: int = 64,
        base_channels: int = 12,
    ) -> None:
        super().__init__()
        if branch_dimension % 4:
            raise ValueError("branch_dimension must be divisible by four")
        self.nostril_encoder = MultiScaleTemporalEncoder(1, branch_dimension, base_channels)
        self.global_encoder = MultiScaleTemporalEncoder(1, branch_dimension, base_channels)
        self.asymmetry_encoder = MultiScaleTemporalEncoder(1, branch_dimension, base_channels)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, branch_dimension))
        self.token_embedding = nn.Parameter(torch.empty(1, 7, branch_dimension))
        nn.init.normal_(self.token_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=branch_dimension,
            nhead=4,
            dim_feedforward=branch_dimension * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(branch_dimension),
            nn.Linear(branch_dimension, embedding_dimension),
            nn.LayerNorm(embedding_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != 2:
            raise ValueError(f"expected (batch, 2, samples), found {tuple(values.shape)}")
        left = values[:, 0:1]
        right = values[:, 1:2]
        h_left = self.nostril_encoder(left)
        h_right = self.nostril_encoder(right)
        tokens = torch.stack(
            (
                h_left,
                h_right,
                torch.abs(h_left - h_right),
                h_left * h_right,
                self.global_encoder(left + right),
                self.asymmetry_encoder(left - right),
            ),
            dim=1,
        )
        summary = self.summary_token.expand(len(values), -1, -1)
        fused = self.fusion(torch.cat((summary, tokens), dim=1) + self.token_embedding)
        return F.normalize(self.projection(fused[:, 0]), dim=1)


class ArcMarginHead(nn.Module):
    """ArcFace classification head used only for training identities."""

    def __init__(
        self,
        embedding_dimension: int,
        classes: int,
        *,
        scale: float = 30.0,
        margin: float = 0.20,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dimension))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.cos_margin = math.cos(margin)
        self.sin_margin = math.sin(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0, 1.0)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.square(), min=1e-7))
        target = cosine * self.cos_margin - sine * self.sin_margin
        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        return self.scale * (one_hot * target + (1.0 - one_hot) * cosine)


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.07
) -> torch.Tensor:
    """Supervised contrastive loss with other windows of an identity as positives."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    if not positive_mask.any(dim=1).all():
        raise ValueError("every anchor requires another positive example in the batch")
    denominator_mask = ~self_mask
    exp_logits = torch.exp(logits) * denominator_mask
    log_probability = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive_log_probability = (
        (positive_mask * log_probability).sum(dim=1) / positive_mask.sum(dim=1)
    )
    return -mean_positive_log_probability.mean()
