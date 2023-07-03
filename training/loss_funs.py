#!/usr/bin/env python3
# -*- coding: utf-8 -*

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float32, PyTree


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class KL_Div:
    type: str

    def tree_flatten(self) -> Tuple[tuple, Dict[str, str]]:
        children = ()
        aux_data = {"type": self.type}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @staticmethod
    def entropy(targets: Float32[Array, "#batch num_cls"]) -> Float32[Array, "#batch"]:
        return jnp.sum(jnp.where(targets == 0, 0, targets * jnp.log(targets)), axis=-1)

    @staticmethod
    def cross_entropy(
        targets: Float32[Array, "#batch num_cls"],
        log_probs: Float32[Array, "#batch num_cls"],
    ) -> Float32[Array, "#batch"]:
        return jnp.sum(targets * log_probs, axis=-1)

    @jax.jit
    def standard_kld(
        self,
        targets: Float32[Array, "#batch num_cls"],
        log_probs: Float32[Array, "#batch num_cls"],
    ) -> Float32[Array, "#batch"]:
        return self.entropy(targets) - self.cross_entropy(targets, log_probs)

    @jax.jit
    def convex_kld(
        self,
        targets: Float32[Array, "#batch num_cls"],
        log_probs: Float32[Array, "#batch num_cls"],
    ) -> Float32[Array, "#batch"]:
        return (
            self.entropy(targets)
            - self.cross_entropy(targets, log_probs)
            - targets
            + jnp.exp(log_probs)
        )

    def kl_divergence(self) -> Callable:
        return getattr(self, f"{self.type}_kld")

    def __call__(
        self,
        targets: Float32[Array, "#batch num_cls"],
        log_probs: Float32[Array, "#batch num_cls"],
    ) -> Float32[Array, "#batch"]:
        return self.kl_divergence(targets, log_probs)


@jax.jit
def accuracy(
    logits: Float32[Array, "#batch num_cls"], targets: Float32[Array, "#batch num_cls"]
) -> Float32[Array, ""]:
    return jnp.mean(
        logits.argmax(axis=1) == jnp.nonzero(targets, size=targets.shape[0])[-1]
    )


@jax.jit
def kl_divergence(
    targets: Float32[Array, "#batch num_cls"],
    log_probs: Float32[Array, "#batch num_cls"],
) -> Float32[Array, "#batch"]:
    kld = targets * (jnp.where(targets == 0, 0, jnp.log(targets)) - log_probs)
    kld = kld - targets + jnp.exp(log_probs)
    return jnp.sum(kld, axis=-1)


def class_hits(
    logits: Float32[Array, "#batch num_cls"],
    targets: Float32[Array, "#batch num_cls"],
    target_type: str,
) -> Dict[int, List[int]]:
    """Compute the per-class accuracy for imbalanced datasets."""
    if isinstance(logits, tuple):
        logits = logits[0]
    if isinstance(targets, tuple):
        targets = targets[0]
    y_hat = logits.argmax(axis=-1)
    cls_hits = defaultdict(list)
    if target_type.startswith("soft"):
        y = targets.argmax(axis=-1)
    else:
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
    loss_p = optax.softmax_cross_entropy(logits[0], y[0]).mean()
    loss_n = optax.softmax_cross_entropy(logits[1], y[1]).mean()
    loss = loss_p + loss_n
    return loss, logits


def loss_fn_resnet(
    state: PyTree,
    params: PyTree[Float32[Array, "..."]],
    X: Float32[Array, "#batchk h w c"],
    y: Float32[Array, "#batch num_cls"],
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    logits, new_state = resnet_predict(state, params, X, train)
    loss_p = optax.softmax_cross_entropy(logits[0], y[0]).mean()
    loss_n = optax.softmax_cross_entropy(logits[1], y[1]).mean()
    loss = loss_p + loss_n
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
    loss_p = optax.softmax_cross_entropy(logits[0], y[0]).mean()
    loss_n = optax.softmax_cross_entropy(logits[1], y[1]).mean()
    loss = loss_p + loss_n
    aux = (logits, rng)
    return loss, aux
