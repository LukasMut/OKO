#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["TripletHead"]

import jax
import flax.linen as nn
import jax.numpy as jnp

from einops import rearrange
from jax import vmap, lax
from functools import partial
from typing import Any
from .modules import Normalization, Sigmoid, Tanh, AttentionBlock

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
        # x = nn.relu(x)
        return x


class TripletHead(nn.Module):
    backbone: str
    triplet_dim: int
    capture_intermediates: bool = False
    dtype: Any = jnp.float32

    def setup(self):

        self.input_layer = nn.Dense(self.triplet_dim * 2)
        self.transformer = AttentionBlock(
            self.triplet_dim * 2, self.triplet_dim * 2, 1, 0.2
        )
        self.dropout = nn.Dropout(0.2)

        # Parameters/Embeddings
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.triplet_dim * 2),
        )
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, 1 + 64, self.triplet_dim * 2),
        )

        self.normalization = Normalization()
        self.triplet_bottleneck = TripletBottleneck(
            backbone=self.backbone,
            width=self.triplet_dim * 3,
            capture_intermediates=self.capture_intermediates,
        )

        if self.backbone.lower() == "vit":
            self.mlp_head = nn.Sequential(
                [
                    nn.LayerNorm(),
                    nn.Dense(3, dtype=self.dtype),
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

    def individual_token_prediction(self, x: Array, k: int = 3) -> Array:
        """For each image in an image triplet, predict individually whether an image is the odd-one-out."""
        assert (
            x.shape[1] == k
        ), f"\nTo predict the odd-one-out from individual tokens, {k:02d} token embeddings are required.\n"
        x = nn.LayerNorm()(x)
        outputs = [nn.Dense(1, dtype=self.dtype)(x[:, i, :]) for i in range(k)]
        outputs = jnp.concatenate(outputs, axis=1)
        return outputs

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = rearrange(x, "(b t) d -> b t d", b=x.shape[0] // 3, t=3)
        # Preprocess input
        B, T, _ = x.shape
        x = self.input_layer(x)

        # Add CLS token and positional encoding
        # cls_token = self.cls_token.repeat(B, axis=0)
        # x = jnp.concatenate([cls_token, x], axis=1)
        # x = x + self.pos_embedding[:, : T + 1]
        x = x + self.pos_embedding[:, :T]

        # Apply Transforrmer
        x = self.dropout(x, deterministic=False)
        x = self.transformer(x)
        x = x[:, 1:, :]

        # print(x.shape)
        # out = self.individual_token_prediction(x)
        # print(out.shape)
        # print()
        # raise Exception

        # x = x[:, 0]

        # x = x.reshape(-1, x.shape[-1])
        # x = self.tripletize(x)(self.trange(x.shape[0]))

        # print(x.shape)
        
        # x = rearrange(x, "b t d -> (b t) d")
        # x = self.tripletize(x)(self.trange(x.shape[0]))
        x = rearrange(x, "b t d -> b (t d)")
        x = self.triplet_bottleneck(x)
        
        # x = rearrange(x, "b (t d) -> b t d", b=x.shape[0], t=3)
        # out = self.individual_token_prediction(x)

        # out = self.mlp_head(x)

        x = nn.LayerNorm()(x)
        out = nn.Dense(3, dtype=self.dtype)(x)

        return out
