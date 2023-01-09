#!/usr/bin/env python3
# -*- coding: utf-8 -*

from collections import defaultdict
from functools import partial
from typing import Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jax import random
from jaxtyping import Array, Float32, PyTree


@jax.jit
def accuracy(
    logits: Float32[Array, "#batch num_cls"], targets: Float32[Array, "#batch num_cls"]
) -> Float32[Array, ""]:
    return jnp.mean(
        logits.argmax(axis=1) == jnp.nonzero(targets, size=targets.shape[0])[-1]
    )


def class_hits(
    logits: Float32[Array, "#batch num_cls"], targets: Float32[Array, "#batch num_cls"]
) -> Dict[int, List[int]]:
    """Compute the per-class accuracy for imbalanced datasets."""
    y_hat = logits.argmax(axis=-1)
    cls_hits = defaultdict(list)
    y = jnp.nonzero(targets, size=targets.shape[0])[-1]
    for i, y_i in enumerate(y):
        cls_hits[y_i.item()].append(1 if y_i == y_hat[i] else 0)
    return cls_hits


@partial(jax.jit, static_argnames=["lmbda"])
def l2_reg(
    params: PyTree[Float32[Array, "..."]], lmbda: float = 1e-3
) -> Float32[Array, ""]:
    """l2 weight regularization during (triplet) pretraining."""
    # NOTE: sum(x ** 2) = ||x||_{2}^{2}
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = lmbda * 0.5 * weight_l2
    return weight_penalty


def custom_predict(
    state: PyTree,
    params: PyTree[Float32[Array, "..."]],
    X: Float32[Array, "#batchk h w c"],
    train: bool = True,
) -> Array:
    return state.apply_fn({"params": params}, X, train=train)


def resnet_predict(
    state: PyTree,
    params: PyTree[Float32[Array, "..."]],
    X: Float32[Array, "#batchk h w c"],
    train: bool,
) -> Union[Tuple[Array, PyTree], Array]:
    return state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        X,
        mutable=["batch_stats"] if train else False,
        train=train,
    )


def vit_predict(
    state: PyTree,
    params: PyTree[Float32[Array, "..."]],
    X: Float32[Array, "#batchk h w c"],
    rng: Array,
    train: bool,
) -> Tuple[Array, Array]:
    rng, dropout_apply_rng = random.split(rng)
    logits = state.apply_fn(
        {"params": params},
        X,
        train=train,
        rngs={"dropout": dropout_apply_rng},
    )
    return logits, rng


def loss_fn_custom(
    state: PyTree,
    params: PyTree[Float32[Array, "..."]],
    X: Float32[Array, "#batchk h w c"],
    y: Float32[Array, "#batch num_cls"],
) -> Tuple[Array, Tuple[Array]]:
    logits = custom_predict(state, params, X)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    return loss, logits


def loss_fn_resnet(
    state: PyTree,
    params: PyTree[Float32[Array, "..."]],
    X: Float32[Array, "#batchk h w c"],
    y: Float32[Array, "#batch num_cls"],
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    logits, new_state = resnet_predict(state, params, X, train)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    aux = (logits, new_state)
    return loss, aux


def loss_fn_vit(
    state: PyTree,
    params: PyTree[Float32[Array, "..."]],
    X: Float32[Array, "#batchk h w c"],
    y: Float32[Array, "#batch num_cls"],
    rng=None,
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    logits, rng = vit_predict(state, params, X, rng, train)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    aux = (logits, rng)
    return loss, aux
