#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    task: str
    num_classes: int = None
    triplet_dim: int = None
    capture_intermediates: bool = False

    def setup(self):
        self.encoder = CNN(
            features=self.encoder_widths,
            source=self.source,
            capture_intermediates=self.capture_intermediates,
        )

        self.mlp_head = self.make_head()

    @nn.nowrap
    def make_head(self):
        """Create target task specific MLP head."""
        if self.task == "ooo_clf":
            assert isinstance(
                self.triplet_dim, int
            ), "\nDimensionality of triplet head bottleneck required.\n"
            head = TripletHead(
                backbone="custom",
                triplet_dim=self.triplet_dim,
                capture_intermediates=self.capture_intermediates,
            )
        elif self.task == "ooo_dist":
            # identity function
            head = nn.Sequential(
                [
                    Normalization(),
                    Identity(),
                ],
            )
        elif self.task.startswith("mle"):
            assert isinstance(
                self.num_classes, int
            ), "\nNumber of classes in dataset required.\n"
            head = nn.Dense(self.num_classes, name="mlp_head")
        else:
            raise ValueError(
                f"\nOutput heads implemented only for the following tasks: {TASKS}.\n"
            )
        return head

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.encoder(x)
        if self.capture_intermediates:
            self.sow("intermediates", "latent_reps")
        out = self.mlp_head(x)
        return out
