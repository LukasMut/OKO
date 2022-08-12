#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataLoader"]

import copy
import math
import random
import jax

import numpy as np
import jax.numpy as jnp
import haiku as hk

from ml_collections import config_dict
from dataclasses import dataclass
from einops import rearrange
from jax import vmap
from functools import partial
from typing import Any, Iterator, List, Tuple


Array = Any
FrozenDict = config_dict.FrozenConfigDict


@dataclass
class DataLoader:
    data: Tuple[Array, Array]
    data_config: FrozenDict
    model_config: FrozenDict
    seed: int
    train: bool = True
    class_subset: List[int] = None

    def __post_init__(self):
        self.cpu_devices = jax.devices("cpu")
        self.num_gpus = 2
        self.device_num = random.choices(
            range(self.num_gpus))[0]
        self.X = self.data[0]
        self.class_subset = copy.deepcopy(self.class_subset)
        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.data_config.name.endswith("mnist"):
            self.X = jnp.expand_dims(self.X, axis=-1)

        if self.model_config.task.startswith("ooo"):
            self.rng_seq = hk.PRNGSequence(self.seed)
            self.y_prime = jnp.nonzero(self.data[1])[-1]
            self.classes = np.unique(self.y_prime)
            self.n_batches = math.ceil(
                self.data_config.max_triplets / self.data_config.batch_size
            )
        else:
            self.y = copy.deepcopy(self.data[1])
            self.dataset = list(zip(self.X, self.y))
            self.n_batches = math.ceil(len(self.dataset) / self.data_config.batch_size)

            if self.data_config.sampling == "uniform":
                num_classes = self.y.shape[-1]
                self.y_flat = np.nonzero(self.data[1])[1]
                if self.class_subset:
                    self.classes = np.array(self.class_subset)
                else:
                    self.classes = np.arange(num_classes)
            else:
                self.remainder = len(self.dataset) % self.data_config.batch_size

        self.create_functions()

    def create_functions(self):
        """
        Create nested functions within class-method to jit them afterwards.
        Note that class-methods cannot be jitted because an object cannot be jitted.
        """

        @jax.jit
        def sample_pairs(seed: int) -> Array:
            """Sample pairs of objects from the same class."""
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, num=self.data_config.batch_size)

            def sample_pair(classes, key):
                return jax.random.choice(key, classes, shape=(2,), replace=False)

            return vmap(partial(sample_pair, self.classes))(keys)

        @partial(jax.jit, static_argnames=["alpha"])
        def label_smoothing(y, alpha: float = 0.1) -> Array:
            """Apply label smoothing."""
            return y * (1 - alpha) + (alpha / y.shape[-1])

        def unzip_pairs(
            dataset: Array,
            subset: range,
            sampling: str,
            train: bool,
            random_order=None,
        ) -> Tuple[Array, Array]:
            """Create tuples of data pairs (X, y)."""
            X, y = zip(
                *[
                    dataset[random_order[i]]
                    if (sampling == "standard" and train)
                    else self.dataset[i]
                    for i in subset
                ]
            )
            X = jnp.stack(X, axis=0)
            y = jnp.stack(y, axis=0)
            return (X, y)

        # create jitted versions of the functions defined above
        self.sample_pairs = sample_pairs
        self.label_smoothing = label_smoothing

        if self.model_config.task.startswith("mle"):
            self.unzip_pairs = partial(unzip_pairs, self.dataset)

    def stepping(self, random_order: Array) -> Iterator:
        """Step over the entire training data in mini-batches of size B."""
        for i in range(self.n_batches):
            if self.remainder != 0 and i == int(self.n_batches - 1):
                subset = range(
                    i * self.data_config.batch_size,
                    i * self.data_config.batch_size + self.remainder,
                )
            else:
                subset = range(
                    i * self.data_config.batch_size,
                    (i + 1) * self.data_config.batch_size,
                )
            X, y = self.unzip_pairs(
                subset=subset,
                sampling=self.data_config.sampling,
                train=self.train,
                random_order=random_order,
            )
            yield (X, y)

    def batch_balancing(self) -> Iterator:
        """Sample classes uniformly for each randomly sampled mini-batch."""
        for _ in range(self.n_batches):
            sample = np.random.choice(self.classes, size=self.data_config.batch_size)
            subset = [
                np.random.choice(np.where(self.y_flat == cls)[0]) for cls in sample
            ]
            X, y = self.unzip_pairs(
                subset=subset,
                sampling=self.data_config.sampling,
                train=self.train,
            )
            yield (X, y)

    @staticmethod
    def expand(pairs: Array) -> Tuple[Array, Array]:
        doubles = np.apply_along_axis(np.random.choice, axis=1, arr=pairs)
        triplets = np.c_[pairs, doubles]
        ooo = np.array(
            [
                np.where(triplet != double)[0][0]
                for triplet, double in zip(triplets, doubles)
            ]
        )
        return triplets, ooo

    def sample_triplets(self, triplets: Array) -> Array:
        def sample_triplet(y_prime, triplet: Array) -> list:
            return [np.random.choice(np.where(y_prime == cls)[0]) for cls in triplet]

        return np.apply_along_axis(
            partial(sample_triplet, self.y_prime), arr=triplets, axis=1
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["last_idx"])
    def convert_labels(labels: Array, last_idx: int = 2) -> Array:
        """Labels for cross-entropy and classification error are rotations of each other."""
        first_conversion = jnp.where(labels != 1, labels - last_idx, labels)
        converted_labels = jnp.where(first_conversion < 0, last_idx, first_conversion)
        return converted_labels

    def ooo_batch_balancing(self) -> Iterator:
        for _ in range(self.n_batches):
            seed = np.random.randint(low=0, high=1e9, size=1)[0]
            pairs_subset = self.sample_pairs(seed)
            triplet_subset, ooo_subset = self.expand(pairs_subset)
            if self.model_config.task == "ooo_dist":
                # centropy and classification error labels are rotations of each other
                ooo_subset = self.convert_labels(ooo_subset)
            y = jax.nn.one_hot(x=ooo_subset, num_classes=3)
            # move NumPy array to device (convert to DeviceArray)
            triplet_subset = self.sample_triplets(triplet_subset)
            # .ravel() is more memory-efficient than .flatten() or .reshape(-1)
            triplet_subset = triplet_subset.ravel()
            X = self.X[triplet_subset]
            if self.model_config.task == "ooo_clf":
                X = rearrange(X, "(n k) h w c -> n k h w c", n=X.shape[0] // 3)
            # X = jax.device_put(X, self.cpu_devices[0])
            # y = jax.device_put(y, self.cpu_devices[0])
            X = jax.device_put(X)#, jax.devices())#[self.device_num])
            y = jax.device_put(y)#, jax.devices())#[self.device_num])
            yield (X, y)

    def __iter__(self) -> Iterator:
        if self.model_config.task.startswith("ooo"):
            return self.ooo_batch_balancing()
        else:
            if self.data_config.sampling == "standard":
                if self.train:
                    # randomly permute the order of samples in the data (i.e., for each epoch shuffle the data)
                    random_order = np.random.permutation(np.arange(len(self.dataset)))
                return self.stepping(random_order)
            else:
                return self.batch_balancing()

    def __len__(self) -> int:
        return self.n_batches
