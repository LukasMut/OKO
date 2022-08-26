#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["TripletHead"]

import jax
import flax.linen as nn
import jax.numpy as jnp

from jax import vmap, lax
from functools import partial
from typing import Any
from einops import rearrange

Array = jnp.ndarray


class TripletBottleneck(nn.Module):
    backbone: str
    width: int
    capture_intermediates: bool

    def setup(self):
        if self.backbone.lower() == "vit":
            self.bottleneck = nn.Sequential(
                [
                    nn.LayerNorm(),
                    nn.Dense(self.width),
                ],
                name="triplet_bottleneck",
            )
        else:
            self.bottleneck = nn.Dense(self.width, name="triplet_bottleneck")

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = self.bottleneck(x)
        if self.capture_intermediates:
            self.sow("intermediates", "representations", x)
        x = nn.relu(x)
        return x


class TripletHead(nn.Module):
    backbone: str
    triplet_dim: int
    capture_intermediates: bool = False
    dtype: Any = jnp.float32

    def setup(self):
        self.triplet_bottleneck = TripletBottleneck(
            backbone=self.backbone,
            width=self.triplet_dim,
            capture_intermediates=self.capture_intermediates,
        )
        if self.backbone.lower() == "vit":
            self.mlp_head = nn.Sequential(
                [
                    nn.LayerNorm(),
                    nn.Dense(3),
                ],
                name="triplet_head",
            )
        else:
            self.mlp_head = nn.Dense(3, dtype=self.dtype)

        # create jitted helper functions
        self.create_functions()

    def create_functions(self, k: int = 3):
        @nn.nowrap
        def tripletize(k: int, x: Array) -> Array:
            """Create triplets of the latent representations."""

            @partial(jax.jit, static_argnames=["k"])
            def concat(k: int, i: int) -> Array:
                triplet = lax.dynamic_slice(x, (i * k, 0), (k, x.shape[1]))
                return jnp.concatenate(triplet, axis=-1)

            return vmap(partial(concat, k))

        def trange(k: int, B: int) -> Array:
            return jnp.arange(0, B // k)

        self.tripletize = partial(tripletize, k)
        self.trange = partial(trange, k)

    
    def individual_prediction(self, x: Array, k: int = 3) -> Array:
        """For each image in an image triplet, predict individually whether an image is the odd-one-out."""
        x = rearrange(x, "(n k) d -> n k d", n=x.shape[0] // k, k=k)
        outputs = [nn.Dense(1, dtype=self.dtype)(x[:, i, :]) for i in range(k)]
        outputs = jnp.concatenate(outputs, axis=1)
        return outputs

    @nn.compact
    def __call__(self, x: Array) -> Array:
        # x = self.tripletize(x)(self.trange(x.shape[0]))
        # x = self.triplet_bottleneck(x)
        # out = self.mlp_head(x)
        out = self.individual_prediction(x)
        return out
