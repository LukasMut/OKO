#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Num
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence
from .triplet import TripletHead

from .modules import Identity, Normalization, Sigmoid
from utils import TASKS

Array = jnp.ndarray


class ConvBlock(nn.Module):
    feat: int
    layer: int
    source: str

    @nn.compact
    def __call__(self, x: Array) -> Array:
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

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i, feat in enumerate(self.features[:-1], start=1):
            x = ConvBlock(feat, i, self.source)(x)
        x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features[-1:], start=i + 1):
            x = FC(feat, i, self.capture_intermediates)(x)
        return x


class Custom(nn.Module):
    encoder_widths: Sequence[int]
    source: str
    num_classes: int
    capture_intermediates: bool = False

    def setup(self):
        self.encoder = CNN(
            features=self.encoder_widths,
            source=self.source,
            capture_intermediates=self.capture_intermediates,
        )
        self.head = TripletHead(
            backbone="custom",
            num_classes=self.num_classes,
        )

    @nn.compact
    def __call__(self, x: Array, train:bool=True) -> Array:
        x = self.encoder(x)
        if self.capture_intermediates:
            self.sow("intermediates", "latent_reps")
        out = self.head(x, train)
        return out
