#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["TripletHead"]

import jax
import flax.linen as nn
import jax.numpy as jnp

from jax import vmap
from typing import Any
from einops import rearrange

Array = jnp.ndarray

class TripletHead(nn.Module):
    backbone: str
    num_classes: int
    k: int = 3
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
    
    def attention(self, x: Array) -> Array:
        dots = vmap(self.query, in_axes=1, out_axes=1)(x)
        # return dots / jnp.sqrt(x.shape[-1])
        return dots

    @nn.compact
    def __call__(self, x: Array, train: bool = True) -> Array:
        if train:
            x = rearrange(x, "(n k) d -> n k d", n=x.shape[0] // self.k, k=self.k)
            dots = self.attention(x)
            out = dots.sum(axis=1)
        else:
            out = self.query(x)
        return out
