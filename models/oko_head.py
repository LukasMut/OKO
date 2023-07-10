#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["OKOHead"]

from typing import Any, Tuple, Union

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
from jax import vmap
from jaxtyping import Array, Float32, jaxtyped
from typeguard import typechecked as typechecker


class OKOHead(nn.Module):
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
            out = self.aggregation(x)
        else:
            out = self.query(x)
        return out
