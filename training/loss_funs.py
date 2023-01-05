#!/usr/bin/env python3
# -*- coding: utf-8 -*

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from jax import random

Array = jnp.ndarray
State = Any
FrozenDict = flax.core.frozen_dict.FrozenDict


@jax.jit
def accuracy(logits: Array, targets: Array) -> Array:
    return jnp.mean(
        logits.argmax(axis=1) == jnp.nonzero(targets, size=targets.shape[0])[-1]
    )


def class_hits(logits: Array, targets: Array) -> Dict[int, List[int]]:
    """Compute the per-class accuracy for imbalanced datasets."""
    y_hat = logits.argmax(axis=-1)
    cls_hits = defaultdict(list)
    y = jnp.nonzero(targets, size=targets.shape[0])[-1]
    for i, y_i in enumerate(y):
        cls_hits[y_i.item()].append(1 if y_i == y_hat[i] else 0)
    return cls_hits


@jax.jit
def l2_reg(params: FrozenDict, lmbda: float = 1e-3) -> float:
    """l2 weight regularization during (triplet) pretraining."""
    # NOTE: sum(x ** 2) = ||x||_{2}^{2}
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = lmbda * 0.5 * weight_l2
    return weight_penalty


def cnn_predict(
    state: FrozenDict, params: FrozenDict, X: Array, train: bool = True
) -> Array:
    return state.apply_fn({"params": params}, X, train=train)


def resnet_predict(
    state: FrozenDict,
    params: FrozenDict,
    X: Array,
    train: bool,
) -> Tuple[Array, State]:
    if train:
        logits, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            X,
            mutable=["batch_stats"] if train else False,
            train=train,
        )
        return logits, new_state
    else:
        logits = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            X,
            mutable=["batch_stats"] if train else False,
            train=train,
        )
        return logits


def vit_predict(
    state: FrozenDict,
    params: FrozenDict,
    rng: Array,
    X: Array,
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


@jax.jit
def loss_fn_custom(
    state: FrozenDict,
    params: FrozenDict,
    X: Array,
    y: Array,
) -> Tuple[Array, Tuple[Array]]:
    logits = cnn_predict(state=state, params=params, X=X)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    return loss, logits


def loss_fn_resnet(
    state: FrozenDict,
    params: FrozenDict,
    X: Array,
    y: Array,
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    logits, new_state = resnet_predict(state=state, params=params, X=X, train=train)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    aux = (logits, new_state)
    return loss, aux


def loss_fn_vit(
    state: FrozenDict,
    params: FrozenDict,
    X: Array,
    y: Array,
    rng=None,
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    logits, rng = vit_predict(state=state, params=params, rng=rng, X=X, train=train)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    aux = (logits, rng)
    return loss, aux
