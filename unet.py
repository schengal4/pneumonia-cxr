import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import List, Optional


class Conv2dAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        norm_layer: str = "bn",
        num_groups: int = 32,  # for GroupNorm,
        activation: str = "ReLU",
        inplace: bool = True,  # for activation
    ):
        if norm_layer == "bn":
            NormLayer = nn.BatchNorm2d
        elif norm_layer == "gn":
            NormLayer = partial(nn.GroupNorm, num_groups=num_groups)
        else:
            raise Exception(
                f"`norm_layer` must be one of [`bn`, `gn`], got `{norm_layer}`"
            )
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm = NormLayer(out_channels)
        self.act = getattr(nn, activation)(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SCSEModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        activation: str = "ReLU",
        inplace: bool = False,
    ):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            getattr(nn, activation)(inplace=inplace),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
        )
        self.sSE = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x).sigmoid() + x * self.sSE(x).sigmoid()


class Attention(nn.Module):
    def __init__(self, name: str, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_layer: str = "bn",
        activation: str = "ReLU",
        attention_type: Optional[str] = None,
    ):
        super().__init__()
        self.conv1 = Conv2dAct(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = Conv2dAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if skip is not None:
            h, w = skip.shape[2:]
            x = F.interpolate(x, size=(h, w), mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        else:
            x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: str = "bn",
        activation: str = "ReLU",
    ):
        conv1 = Conv2dAct(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
        )
        conv2 = Conv2dAct(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer=norm_layer,
            activation=activation,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        decoder_n_blocks: int,
        decoder_channels: List[int],
        encoder_channels: List[int],
        decoder_center_block: bool = False,
        decoder_norm_layer: str = "bn",
        decoder_attention_type: Optional[str] = None,
    ):
        super().__init__()

        self.decoder_n_blocks = decoder_n_blocks
        self.decoder_channels = decoder_channels
        self.encoder_channels = encoder_channels
        self.decoder_center_block = decoder_center_block
        self.decoder_norm_layer = decoder_norm_layer
        self.decoder_attention_type = decoder_attention_type

        if self.decoder_n_blocks != len(self.decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    self.decoder_n_blocks, len(self.decoder_channels)
                )
            )
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(self.decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = self.decoder_channels

        if self.decoder_center_block:
            self.center = CenterBlock(
                head_channels, head_channels, norm_layer=self.decoder_norm_layer
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(
            norm_layer=self.decoder_norm_layer,
            attention_type=self.decoder_attention_type,
        )
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        output = [self.center(head)]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            output.append(decoder_block(output[-1], skip))

        return output


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.drop = nn.Dropout2d(p=dropout)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        if isinstance(size, (tuple, list)):
            self.up = nn.Upsample(size=size, mode="bilinear")
        else:
            self.up = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.conv(self.drop(x)))
