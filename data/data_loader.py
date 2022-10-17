#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataLoader"]

import copy
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import vmap
from ml_collections import config_dict

Array = jnp.ndarray
FrozenDict = config_dict.FrozenConfigDict


@dataclass(init=True, repr=True)
class DataLoader:
    data: Tuple[Array, Array]
    data_config: FrozenDict
    model_config: FrozenDict
    seed: int
    train: bool = True
    class_subset: List[int] = None

    def __post_init__(self) -> None:
        self.cpu_devices = jax.devices("cpu")
        self.num_gpus = 2
        self.device_num = random.choices(range(self.num_gpus))[0]
        self.X = self.data[0]
        self.y = copy.deepcopy(self.data[1])

        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.data_config.name.endswith("mnist"):
            self.X = jnp.expand_dims(self.X, axis=-1)

        self.num_classes = self.y.shape[-1]
        self.y_prime = jnp.nonzero(self.data[1])[-1]
        self.ooo_classes = np.unique(self.y_prime)

        if self.train:
            self.num_batches = math.ceil(
                self.data_config.max_triplets / self.data_config.ooo_batch_size
            )
        else:
            self.dataset = list(zip(self.X, self.y))
            self.num_batches = math.ceil(
                len(self.dataset) / self.data_config.main_batch_size
            )
            self.remainder = len(self.dataset) % self.data_config.main_batch_size

        self.y_flat = np.nonzero(self.y)[1]

        if self.data_config.sampling == "dynamic":
            occurrences = dict(
                sorted(Counter(self.y_flat.tolist()).items(), key=lambda kv: kv[0])
            )
            self.hist = jnp.array(list(occurrences.values()))
            self.p = self.hist / self.hist.sum()
            self.temperature = 0.1

        self.create_functions()

    def create_functions(self) -> None:
        def sample_double(classes: Array, q: float, key: Array) -> Array:
            return jax.random.choice(key, classes, shape=(2,), replace=False, p=q)

        @partial(jax.jit, static_argnames=["seed"])
        def sample_doubles(seed: int, q=None) -> Array:
            """Sample pairs of objects from the same class."""
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, num=self.data_config.ooo_batch_size)
            return vmap(partial(self.sample_double, q))(keys)

        def unzip_pairs(dataset: Array, subset: range) -> Tuple[Array, Array]:
            """Create tuples of data pairs (X, y)."""
            X, y = zip(*[dataset[i] for i in subset])
            X = jnp.stack(X, axis=0)
            y = jnp.stack(y, axis=0)
            return (X, y)

        # jit or partially initialize functions for computational efficiency
        if self.train:
            self.sample_double = partial(sample_double, self.ooo_classes)
            self.sample_doubles = sample_doubles
        else:
            self.unzip_pairs = partial(unzip_pairs, self.dataset)

    @staticmethod
    def expand(doubles: Array) -> Tuple[Array, Array]:
        pair_classes = np.apply_along_axis(np.random.choice, axis=1, arr=doubles)
        triplets = np.c_[doubles, pair_classes]
        triplets = np.apply_along_axis(np.random.permutation, axis=1, arr=triplets)
        """
        ooo = np.array(
            [
                np.where(triplet != double)[0][0]
                for triplet, double in zip(triplets, pair_classes)
            ]
        )
        return triplets, ooo, pairs
        """
        return triplets, pair_classes

    def sample_triplets(self, triplets: Array) -> Array:
        def sample_triplet(y_prime, triplet: Array) -> List[int]:
            return [np.random.choice(np.where(y_prime == cls)[0]) for cls in triplet]

        return np.apply_along_axis(
            partial(sample_triplet, self.y_prime), arr=triplets, axis=1
        )

    def stepping(self) -> Iterator:
        """Step over the entire training data in mini-batches of size B."""
        for i in range(self.num_batches):
            if self.remainder != 0 and i == int(self.num_batches - 1):
                subset = range(
                    i * self.data_config.main_batch_size,
                    i * self.data_config.main_batch_size + self.remainder,
                )
            else:
                subset = range(
                    i * self.data_config.main_batch_size,
                    (i + 1) * self.data_config.main_batch_size,
                )
            X, y = self.unzip_pairs(subset)
            yield (X, y)

    def sample_ooo_batch(self, q=None) -> Tuple[Array, Array]:
        """Uniformly sample odd-one-out triplet task mini-batches."""
        seed = np.random.randint(low=0, high=1e9, size=1)[0]
        doubles_subset = self.sample_doubles(seed, q=q)
        # NOTE: the two lines below are necessary for performing a triplet odd-one-out task,
        # using indexes rather than classes as possible choices
        # triplet_subset, ooo_subset, pair_classes = self.expand(doubles_subset)
        # y = jax.nn.one_hot(x=ooo_subset, num_classes=3)
        triplet_subset, pair_classes = self.expand(doubles_subset)
        y = jax.nn.one_hot(x=pair_classes, num_classes=self.num_classes)
        triplet_subset = self.sample_triplets(triplet_subset)
        triplet_subset = triplet_subset.ravel()
        X = self.X[triplet_subset]
        X = rearrange(X, "(n k) h w c -> n k h w c", n=X.shape[0] // 3)
        X = jax.device_put(X)
        y = jax.device_put(y)
        return (X, y)

    def smoothing(self) -> Array:
        @jax.jit
        def softmax(p: Array, beta: float) -> Array:
            return jnp.exp(p / beta) / (
                jnp.exp(p / beta).sum()
            )
        return partial(softmax, self.p)(self.temperature)

    def ooo_batch_balancing(self) -> Tuple[Array, Array]:
        """Simultaneously sample odd-one-out triplet and main multi-class task mini-batches."""
        q = self.smoothing() if self.data_config.sampling == "dynamic" else None
        for _ in range(self.num_batches):
            ooo_batch = self.sample_ooo_batch(q)
            yield ooo_batch
        if self.data_config.sampling == "dynamic":
            self.temperature += 0.05

    def __iter__(self) -> Iterator:
        if self.train:
            return iter(self.ooo_batch_balancing())
        return iter(self.stepping())

    def __len__(self) -> int:
        return self.num_batches
