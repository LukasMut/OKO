#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["OKOHead"]

from typing import Any, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, Float32, jaxtyped
from typeguard import typechecked as typechecker


class OKOHead(nn.Module):
    backbone: str
    num_classes: int
    k: int
    features: int
    dtype: Any = jnp.float32

    def setup(self):
        if self.backbone.lower() == "vit":
            self.query = nn.Sequential(
                [
                    nn.LayerNorm(),
                    nn.Dense(self.num_classes),
                ],
                name="oko_query",
            )
            self.key = nn.Sequential(
                [
                    nn.LayerNorm(),
                    nn.Dense(self.num_classes),
                ],
                name="oko_key",
            )
        else:
            self.query = nn.Dense(self.num_classes, name="oko_query")
            self.key = nn.Dense(self.num_classes, name="oko_key")
        self.attention = self.param(
            "attention", jax.nn.initializers.ones, ((self.k + 2) * self.features,)
        )

    @jaxtyped
    @typechecker
    def aggregation(
        self, x: Float32[Array, "#batchk d"]
    ) -> Float32[Array, "#batch num_cls"]:
        """Aggregate logits over all members in each set."""
        x = rearrange(x, "(b k) d -> b k d", b=x.shape[0] // (self.k + 2), k=self.k + 2)
        dots = vmap(self.query, in_axes=1, out_axes=1)(x)
        out = dots.sum(axis=1)
        return out

    @nn.compact
    @jaxtyped
    @typechecker
    def __call__(
        self,
        x: Float32[Array, "#batchk d"],
        train: bool = True,
    ) -> Union[
        Float32[Array, "#batch num_cls"],
        Tuple[Float32[Array, "#batch num_cls"], Float32[Array, "#batch num_cls"]],
    ]:
        if train:
            out_p = self.aggregation(x)
            x = rearrange(
                x, "(b k) d -> b (k d)", b=x.shape[0] // (self.k + 2), k=self.k + 2
            )
            x = x + (x * self.attention)
            out_n = self.key(x)
            out = (out_p, out_n)
            """
            out = out_p
            """
        else:
            out = self.query(x)
        return out
