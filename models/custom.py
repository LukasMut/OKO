#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence, Tuple, Union

import flax.linen as nn
from jaxtyping import AbstractDtype, Array, Float32, jaxtyped
from typeguard import typechecked as typechecker

from models.oko_head import OKOHead


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


class ConvBlock(nn.Module):
    feat: int
    layer: int
    source: str

    @nn.compact
    def __call__(
        self, x: UInt8orFP32[Array, "#batchk h w c"]
    ) -> Float32[Array, "#batchk h w c"]:
        x = nn.Conv(features=self.feat, kernel_size=(3, 3), name=f"layer_{self.layer}")(
            x
        )
        x = nn.relu(x)
        pooling = nn.avg_pool if self.source.lower().endswith("mnist") else nn.max_pool
        x = pooling(x, window_shape=(2, 2), strides=(2, 2))
        return x


class FC(nn.Module):
    feat: int
    layer: int
    capture_intermediates: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Dense(self.feat, name=f"layer_{self.layer}")(x)
        if self.capture_intermediates:
            self.sow("intermediates", "representations", x)
        x = nn.relu(x)
        return x


class CNN(nn.Module):
    features: Sequence[int]
    source: str
    capture_intermediates: bool = False

    def setup(self):
        self.conv_block = ConvBlock
        self.fc = FC

    @nn.compact
    def __call__(
        self, x: UInt8orFP32[Array, "#batchk h w c"]
    ) -> Float32[Array, "#batchk d"]:
        for layer, feat in enumerate(self.features[:-1], start=1):
            x = self.conv_block(feat, layer, self.source)(x)
        x = x.reshape((x.shape[0], -1))
        for layer, feat in enumerate(self.features[-1:], start=layer + 1):
            x = self.fc(feat, layer, self.capture_intermediates)(x)
        return x


class Custom(nn.Module):
    encoder_widths: Sequence[int]
    num_classes: int
    k: int
    source: str
    capture_intermediates: bool = False

    def setup(self):
        self.encoder = CNN(
            features=self.encoder_widths,
            source=self.source,
            capture_intermediates=self.capture_intermediates,
        )
        self.head = OKOHead(
            backbone="custom",
            num_classes=self.num_classes,
            k=self.k,
        )

    @nn.compact
    @jaxtyped
    @typechecker
    def __call__(
        self, x: UInt8orFP32[Array, "#batchk h w c"], train: bool = True
    ) -> Union[
        Float32[Array, "#batch num_cls"],
        Tuple[Float32[Array, "#batch num_cls"], Float32[Array, "#batch num_cls"]],
    ]:
        x = self.encoder(x)
        out = self.head(x, train)
        return out
