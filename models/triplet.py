#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["TripletHead"]

from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, Float32, jaxtyped
from typeguard import typechecked as typechecker

Array = jnp.ndarray


class TripletHead(nn.Module):
    backbone: str
    num_classes: int
    k: int
    dtype: Any = jnp.float32

    def setup(self):
        if self.backbone.lower() == "vit":
            self.query = nn.Sequential(
                [
                    nn.LayerNorm(),
                    nn.Dense(self.num_classes),
                ],
                name="triplet_query",
            )
        else:
            self.query = nn.Dense(self.num_classes, name="triplet_query")

    @jaxtyped
    @typechecker
    def aggregation(
        self, x: Float32[Array, "#batch k d"]
    ) -> Float32[Array, "#batch num_cls"]:
        """Aggregate logits over all members in a set."""
        dots = vmap(self.query, in_axes=1, out_axes=1)(x)
        # NOTE: scaling/normalizing does not seem to help
        # dots =/ jnp.sqrt(self.num_classes)
        out = dots.sum(axis=1)
        return out

    @nn.compact
    @jaxtyped
    @typechecker
    def __call__(
        self, x: Float32[Array, "#batchk d"], train: bool = True
    ) -> Float32[Array, "#batch num_cls"]:
        if train:
            x = rearrange(
                x, "(n k) d -> n k d", n=x.shape[0] // (self.k + 2), k=self.k + 2
            )
            out = self.aggregation(x)
        else:
            out = self.query(x)
        return out
