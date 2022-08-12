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
def c_entropy(targets, logits, w=None) -> jnp.ndarray:
    if isinstance(w, jnp.ndarray):
        # compute weighted version of the cross-entropy error
        targets *= w
    B = targets.shape[0]
    nll = jnp.sum(-(targets * logits)) / B
    return nll


@jax.jit
def accuracy(logits: Array, targets: Array) -> Array:
    return jnp.mean(
        logits.argmax(axis=1) == jnp.nonzero(targets, size=targets.shape[0])[-1]
    )


@jax.jit
def convert_labels(labels: Array) -> Array:
    first_conversion = jnp.where(labels != 1, labels - 2, labels)
    converted_labels = jnp.where(first_conversion < 0, 2, first_conversion)
    return converted_labels


def ooo_accuracy(logits: Array, targets: Array) -> Array:
    return jnp.mean(
        convert_labels(logits.argmax(axis=1))
        == convert_labels(jnp.nonzero(targets, size=targets.shape[0])[-1])
    )


@jax.jit
def permutation_centropy(logits: Array, y_perms: Array) -> Array:
    centropy = getattr(optax, "softmax_cross_entropy")
    loss = jnp.mean(vmap(lambda y_hat, y: centropy(y_hat, y).mean())(logits, y_perms))
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


def get_triplets(logits: Array) -> Tuple[Array]:
    triplets = logits.reshape(-1, 3, logits.shape[-1])
    feats_i = triplets[:, 0, :]
    feats_j = triplets[:, 1, :]
    feats_k = triplets[:, 2, :]
    return (feats_i, feats_j, feats_k)


# @jax.jit
def get_feature_norms(feats_i: Array, feats_j: Array, feats_k: Array) -> Tuple[Array]:
    feat_i_norms = jnp.linalg.norm(feats_i, ord=2, axis=1)
    feat_j_norms = jnp.linalg.norm(feats_j, ord=2, axis=1)
    feat_k_norms = jnp.linalg.norm(feats_k, ord=2, axis=1)
    return (feat_i_norms, feat_j_norms, feat_k_norms)


@jax.jit
def compute_similarities(
    feats_i: Array, feats_j: Array, feats_k: Array
) -> Tuple[Array]:
    """Apply the similarity function (modeled as the dot product or the cosine similarity) to each pair in the triplet."""
    # feat_norms = get_feature_norms(feats_i, feats_j, feats_k)
    sim_i = jnp.sum(feats_i * feats_j, axis=1)  # / (feat_norms[0] * feat_norms[1])
    sim_j = jnp.sum(feats_i * feats_k, axis=1)  # / (feat_norms[0] * feat_norms[2])
    sim_k = jnp.sum(feats_j * feats_k, axis=1)  # / (feat_norms[1] * feat_norms[2])
    return jnp.stack((sim_i, sim_j, sim_k), axis=1)


@partial(jax.jit, static_argnames=["t"])
def logsumexp(x: Array, t: float) -> Array:
    return jax.nn.logsumexp(a=x, b=1 / t)


def vlog_softmax(batch_similarities: Array, y: Array, t: float = 1.0) -> Array:
    """Vectorized log-softmax function for supervised contrastive loss computation."""

    def log_softmax(t: float, triplet_similarities: Array, y_i: Array):
        return (triplet_similarities[jnp.nonzero(y_i, size=1)[0]] / t) - logsumexp(
            triplet_similarities, t
        )

    return vmap(partial(log_softmax, t))(batch_similarities, y)


@jax.jit
def cross_entropy_loss(batch_sims: Array, y: Array) -> Array:
    """Supervised contrastive loss function."""
    return jnp.mean(-vlog_softmax(batch_sims, y))


@jax.jit
def cnn_predict(state: FrozenDict, params: FrozenDict, X: Array) -> Array:
    return state.apply_fn({"params": params}, X)


def resnet_predict(
    state: FrozenDict,
    params: FrozenDict,
    X: Array,
    train: bool,
) -> Tuple[Array, State]:
    logits, new_state = state.apply_fn(
        {"params": params, "batch_stats": state.batch_stats},
        X,
        mutable=["batch_stats"] if train else False,
    )
    return logits, new_state


def vit_predict(
    state: FrozenDict,
    params: FrozenDict,
    rng: Array,
    X: Array,
    train: bool,
) -> Tuple[Array, Array]:
    rng, dropout_apply_rng = random.split(rng)
    logits = state.apply_fn(
        {"params": params}, X, train=train, rngs={"dropout": dropout_apply_rng}
    )
    return logits, rng


# @jax.jit
def mle_loss_fn_vit(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    rng=None,
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits, rng = vit_predict(state=state, params=params, rng=rng, X=X, train=train)
    loss = optax.softmax_cross_entropy(logits=logits, labels=y).mean()
    aux = (logits, rng)
    return loss, aux


# @jax.jit
def mle_loss_resnet(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits, new_state = resnet_predict(state=state, params=params, X=X, train=train)
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
    logits = cnn_predict(state=state, params=params, X=X)
    loss = optax.softmax_cross_entropy(logits=logits, labels=y).mean()
    return loss, logits


@jax.jit
def ooo_dist_loss_fn_custom(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits = cnn_predict(state=state, params=params, X=X)
    feats_i, feats_j, feats_k = get_triplets(logits)
    batch_similarities = compute_similarities(feats_i, feats_j, feats_k)
    loss = cross_entropy_loss(batch_similarities, y)
    probas = jax.nn.softmax(batch_similarities, axis=1)
    return loss, probas


# @jax.jit
def ooo_dist_loss_fn_resnet(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits, new_state = resnet_predict(state=state, params=params, X=X, train=train)
    feats_i, feats_j, feats_k = get_triplets(logits)
    batch_similarities = compute_similarities(feats_i, feats_j, feats_k)
    loss = cross_entropy_loss(batch_similarities, y)
    probas = jax.nn.softmax(batch_similarities, axis=1)
    aux = (probas, new_state)
    return loss, aux


# @jax.jit
def ooo_dist_loss_fn_vit(
    state: FrozenDict,
    params: FrozenDict,
    batch: Tuple[Array, Array],
    rng=None,
    train: bool = True,
) -> Tuple[Array, Tuple[Array]]:
    """MLE loss function used during finetuning."""
    X, y = batch
    logits, rng = vit_predict(state=state, params=params, rng=rng, X=X, train=train)
    feats_i, feats_j, feats_k = get_triplets(logits)
    batch_similarities = compute_similarities(feats_i, feats_j, feats_k)
    loss = cross_entropy_loss(batch_similarities, y)
    probas = jax.nn.softmax(batch_similarities, axis=1)
    aux = (probas, rng)
    return loss, aux


@jax.jit
def cnn_symmetrize(
    state: FrozenDict, params: FrozenDict, batch: Tuple[Array], p: Array
) -> Tuple[Array]:
    # symmetrize the neural network function
    # to enforce some sort of permutation invariance
    X, y = batch
    triplet_perm = X[:, p, :, :, :]
    triplet_perm = rearrange(triplet_perm, "b k h w c -> (b k) h w c")
    logits = cnn_predict(state, params, triplet_perm)
    return logits, y[:, p]


@jax.jit
def ooo_clf_loss_fn_custom(
    state: FrozenDict,
    perms: Array,
    params: FrozenDict,
    batch: Tuple[Array, Array],
) -> Tuple[Array, Tuple[Array]]:
    """Loss function to predict the odd-one-out in a triplet of images."""
    # TODO: investigate whether two or three permutations work better
    # NOTE: more than two or three permutations are computationally too expensive
    positions = jax.device_put(
        np.random.choice(np.arange(perms.shape[0]), size=2, replace=False)
    )
    logits, y_perms = vmap(partial(cnn_symmetrize, state, params, batch))(
        perms[positions]
    )
    loss = permutation_centropy(logits=logits, y_perms=y_perms)
    acc = permutation_accuracy(logits=logits, y_perms=y_perms)
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
        state=state, params=params, X=triplet_perm, train=train
    )
    return logits, y[:, p], new_state


# @jax.jit
def ooo_clf_loss_fn_resnet(
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
        np.random.choice(np.arange(perms.shape[0]), size=2, replace=False)
    )
    logits, y_perms, new_state = vmap(
        partial(resnet_symmetrize, state, params, batch, train)
    )(perms[positions])
    loss = permutation_centropy(logits=logits, y_perms=y_perms)
    acc = permutation_accuracy(logits=logits, y_perms=y_perms)
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
        state=state, params=params, rng=rng, X=triplet_perm, train=train
    )
    return logits, y[:, p], rng


# @jax.jit
def ooo_clf_loss_fn_vit(
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
        np.random.choice(np.arange(perms.shape[0]), size=2, replace=False)
    )
    logits, y_perms, rng = vmap(
        partial(vit_symmetrize, state, params, batch, rng, train)
    )(perms[positions])
    loss = permutation_centropy(logits=logits, y_perms=y_perms)
    acc = permutation_accuracy(logits=logits, y_perms=y_perms)
    aux = (acc, rng)
    return loss, aux
