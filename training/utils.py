#!/usr/bin/env python3
# -*- coding: utf-8 -*

from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange
from jax import random, vmap

Array = jnp.ndarray
State = Any
FrozenDict = flax.core.frozen_dict.FrozenDict


@jax.jit
def c_entropy(targets, logits, w=None) -> Array:
    if isinstance(w, Array):
        # compute weighted version of the cross-entropy error
        targets *= w
    B = targets.shape[0]
    nll = jnp.sum(-(targets * jax.nn.log_softmax(logits, axis=1))) / B
    return nll


@jax.jit
def accuracy(logits: Array, targets: Array) -> Array:
    return jnp.mean(
        logits.argmax(axis=1) == jnp.nonzero(targets, size=targets.shape[0])[-1]
    )


@jax.jit
def permutation_centropy(logits: Array, y_perms: Array) -> Array:
    centropy = getattr(optax, "softmax_cross_entropy")
    loss = (vmap(lambda y_hat, y: centropy(y_hat, y).mean())(logits, y_perms)).mean()
    return loss


def permutation_accuracy(logits: Array, y_perms: Array) -> Array:
    acc = jnp.mean(vmap(accuracy)(logits, y_perms))
    return acc


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
    weight_penalty_params = jax.tree_leaves(params)
    weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = lmbda * 0.5 * weight_l2
    # weight_l1 = sum(jnp.sum(abs(x)) for x in weight_penalty_params if x.ndim > 1)
    # weight_penalty = lmbda * 0.5 * weight_l1
    return weight_penalty


@jax.jit
def inductive_bias(
    finetuned_params: FrozenDict, pretrained_params: FrozenDict, lmbda: float = 1e-4
) -> float:
    """Keep finetuned params close to pretrained params."""
    # NOTE: sum((x_{i} - x_{j}) ** 2) = ||x_{i} - x_{j}||_{2}^{2}
    pretrained_params = jax.tree_leaves(pretrained_params)
    finetuned_params = jax.tree_leaves(finetuned_params)
    weight_l2 = sum(
        jnp.sum((x_i - x_j) ** 2)
        for x_i, x_j in zip(pretrained_params, finetuned_params)
        if (x_i.ndim > 1 and x_j.ndim > 1)
    )
    bias = lmbda * 0.5 * weight_l2
    return bias


def cnn_predict(
    state: FrozenDict, params: FrozenDict, X: Array, task: str, rng: Array = None
) -> Array:
    """
    if task == 'ooo':
        _, dropout_apply_rng = random.split(rng)
        logits = state.apply_fn(
            {"params": params},
            X,
            task=task,
            rngs={"dropout": dropout_apply_rng},
        )
        return logits
    else:
    """
    return state.apply_fn({"params": params}, X, task=task)


def resnet_predict(
    state: FrozenDict,
    params: FrozenDict,
    X: Array,
    train: bool,
    task: str,
) -> Tuple[Array, State]:
    if train:
        logits, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            X,
            mutable=["batch_stats"] if train else False,
            train=train,
            task=task,
        )
        return logits, new_state
    else:
        logits = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            X,
            mutable=["batch_stats"] if train else False,
            train=train,
            task=task,
        )
        return logits


def vit_predict(
    state: FrozenDict,
    params: FrozenDict,
    rng: Array,
    X: Array,
    train: bool,
    task: str,
) -> Tuple[Array, Array]:
    rng, dropout_apply_rng = random.split(rng)
    logits = state.apply_fn(
        {"params": params},
        X,
        train=train,
        rngs={"dropout": dropout_apply_rng},
        task=task,
    )
    return logits, rng


def mle_loss_fn_vit(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    rng=None,
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits, rng = vit_predict(
        state=state, params=params, rng=rng, X=X, train=train, task="mle"
    )
    loss = optax.softmax_cross_entropy(logits=logits, labels=y).mean()
    aux = (logits, rng)
    return loss, aux


def mle_loss_fn_resnet(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits, new_state = resnet_predict(
        state=state, params=params, X=X, train=train, task="mle"
    )
    loss = optax.softmax_cross_entropy(logits=logits, labels=y).mean()
    aux = (logits, new_state)
    return loss, aux


@jax.jit
def mle_loss_fn_custom(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits = cnn_predict(state=state, params=params, X=X, task="mle")
    loss = optax.softmax_cross_entropy(logits=logits, labels=y).mean()
    return loss, logits


@jax.jit
def cnn_symmetrize(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array],
    rng: Array,
    p: Array,
) -> Tuple[Array]:
    # symmetrize the neural network function
    # to enforce some sort of permutation invariance
    X, y = batch
    triplet_perm = X[:, p, :, :, :]
    triplet_perm = rearrange(triplet_perm, "b k h w c -> (b k) h w c")
    # logits = cnn_predict(state, params, triplet_perm, task="ooo", rng=rng)
    logits = cnn_predict(state, params, triplet_perm, task="ooo")
    return logits, y[:, p]


# @jax.jit
def ooo_loss_fn_custom(
    state: FrozenDict,
    perms: Array,
    params: FrozenDict,
    batch: Tuple[Array, Array],
) -> Tuple[Array, Tuple[Array]]:
    """Loss function to predict the odd-one-out in a triplet of images."""
    # TODO: investigate whether two or three permutations work better
    # NOTE: more than two or three permutations are computationally too expensive
    positions = jax.device_put(
        np.random.choice(np.arange(perms.shape[0]), size=6, replace=False)
    )
    rng = jax.random.PRNGKey(np.random.randint(low=0, high=1e9, size=1)[0])
    logits, y_perms = vmap(partial(cnn_symmetrize, state, params, batch, rng))(
        perms[positions]
    )

    loss = permutation_centropy(logits=logits, y_perms=y_perms)
    acc = permutation_accuracy(logits=logits, y_perms=y_perms)

    """
    loss = optax.softmax_cross_entropy(logits, y_perms).mean()
    acc = accuracy(logits, y_perms)
    """
    return loss, (acc)


def resnet_symmetrize(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array],
    train: bool,
    p: Array,
) -> Tuple[Array, Array, State]:
    # symmetrize ResNet to enforce permutation invariance
    X, y = batch
    triplet_perm = X[:, p, :, :, :]
    triplet_perm = rearrange(triplet_perm, "b k h w c -> (b k) h w c")
    logits, new_state = resnet_predict(
        state=state, params=params, X=triplet_perm, train=train, task="ooo"
    )
    return logits, y[:, p], new_state


def ooo_loss_fn_resnet(
    state: FrozenDict,
    perms: Array,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """Loss function to predict the odd-one-out in a triplet of images."""
    # TODO: investigate whether two or three permutations work better
    # NOTE: more than two or three permutations are computationally too expensive
    positions = jax.device_put(
        np.random.choice(np.arange(perms.shape[0]), size=1, replace=False)
    )
    logits, y_perms, new_state = vmap(
        partial(resnet_symmetrize, state, params, batch, train)
    )(perms[positions])
    """
    loss = permutation_centropy(logits=logits, y_perms=y_perms)
    acc = permutation_accuracy(logits=logits, y_perms=y_perms)
    """
    loss = optax.softmax_cross_entropy(logits, y_perms).mean()
    acc = accuracy(logits, y_perms)
    aux = (acc, new_state)
    return loss, aux


def vit_symmetrize(
    state: FrozenDict,
    params: FrozenDict,
    batch: Array,
    rng: Array,
    train: bool,
    p: Array,
) -> Tuple[Array, Array, Array]:
    # symmetrize ViT to enforce permutation invariance
    X, y = batch
    triplet_perm = X[:, p, :, :, :]
    triplet_perm = rearrange(triplet_perm, "b k h w c -> (b k) h w c")
    logits, rng = vit_predict(
        state=state,
        params=params,
        rng=rng,
        X=triplet_perm,
        train=train,
        task="ooo",
    )
    return logits, y[:, p], rng


def ooo_loss_fn_vit(
    state: FrozenDict,
    perms: Array,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    rng=None,
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """Loss function to predict the odd-one-out in a triplet of images."""
    # TODO: investigate whether two or three permutations work better
    # NOTE: more than two or three permutations are computationally too expensive
    positions = jax.device_put(
        np.random.choice(np.arange(perms.shape[0]), size=1, replace=False)
    )
    logits, y_perms, rng = vmap(
        partial(vit_symmetrize, state, params, batch, rng, train)
    )(perms[positions])
    """
    loss = permutation_centropy(logits=logits, y_perms=y_perms)
    acc = permutation_accuracy(logits=logits, y_perms=y_perms)
    """
    loss = optax.softmax_cross_entropy(logits, y_perms).mean()
    acc = accuracy(logits, y_perms)
    aux = (acc, rng[0])
    return loss, aux
