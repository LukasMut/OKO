#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataLoader"]

import copy
import math
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterator, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from jax import vmap
from ml_collections import config_dict

Array = jnp.ndarray
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
        self.device_num = random.choices(range(self.num_gpus))[0]
        self.X = self.data[0]
        self.class_subset = copy.deepcopy(self.class_subset)

        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.data_config.name.endswith("mnist"):
            self.X = jnp.expand_dims(self.X, axis=-1)

        if self.model_config.task == "mtl":
            # variables for main classification task
            self.y = copy.deepcopy(self.data[1])
            self.dataset = list(zip(self.X, self.y))
            self.main_batch_size = self.data_config.batch_size
            num_classes = self.y.shape[-1]
            self.y_flat = np.nonzero(self.data[1])[1]
            self.main_classes = np.arange(num_classes)
            # variables for odd-one-out auxiliary task
            self.rng_seq = hk.PRNGSequence(self.seed)
            self.y_prime = jnp.nonzero(self.data[1])[-1]
            self.ooo_classes = np.unique(self.y_prime)
            # TODO: figure out whether it's useful to use larger batch sizes for the odd-one-out task
            self.ooo_batch_size = self.data_config.batch_size
            self.num_batches = math.ceil(len(self.dataset) / self.main_batch_size)
        else:
            # variables for main classification task
            self.y = copy.deepcopy(self.data[1])
            self.dataset = list(zip(self.X, self.y))
            self.main_batch_size = self.data_config.batch_size
            self.num_batches = math.ceil(len(self.dataset) / self.main_batch_size)

            if self.data_config.sampling == "uniform":
                num_classes = self.y.shape[-1]
                self.y_flat = np.nonzero(self.data[1])[1]

                if self.class_subset:
                    self.main_classes = np.array(self.class_subset)
                else:
                    self.main_classes = np.arange(num_classes)
            else:
                self.remainder = len(self.dataset) % self.main_batch_size

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
            keys = jax.random.split(key, num=self.ooo_batch_size)

            def sample_pair(classes, key):
                return jax.random.choice(key, classes, shape=(2,), replace=False)

            return vmap(partial(sample_pair, self.ooo_classes))(keys)

        @partial(jax.jit, static_argnames=["alpha"])
        def label_smoothing(y, alpha: float = 0.1) -> Array:
            """Apply label smoothing to the original labels."""
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

        # jit functions for computational efficiency
        self.sample_pairs = sample_pairs
        self.label_smoothing = label_smoothing
        self.unzip_pairs = partial(unzip_pairs, self.dataset)

    def stepping(self, random_order: Array) -> Tuple[Array, Array]:
        """Step over the entire training data in mini-batches of size B."""
        for i in range(self.num_batches):
            if self.remainder != 0 and i == int(self.num_batches - 1):
                subset = range(
                    i * self.main_batch_size,
                    i * self.main_batch_size + self.remainder,
                )
            else:
                subset = range(
                    i * self.main_batch_size,
                    (i + 1) * self.main_batch_size,
                )
            X, y = self.unzip_pairs(
                subset=subset,
                sampling=self.data_config.sampling,
                train=self.train,
                random_order=random_order,
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

    def sample_main_batch(self) -> Tuple[Array, Array]:
        sample = np.random.choice(self.main_classes, size=self.main_batch_size)
        subset = [np.random.choice(np.where(self.y_flat == cls)[0]) for cls in sample]
        X, y = self.unzip_pairs(
            subset=subset,
            sampling=self.data_config.sampling,
            train=self.train,
        )
        return (X, y)

    def sample_ooo_batch(self) -> Tuple[Array, Array]:
        """Uniformly sample odd-one-out triplet task mini-batches."""
        seed = np.random.randint(low=0, high=1e9, size=1)[0]
        pairs_subset = self.sample_pairs(seed)
        triplet_subset, ooo_subset = self.expand(pairs_subset)
        y = jax.nn.one_hot(x=ooo_subset, num_classes=3)
        # move NumPy array to device (convert to DeviceArray)
        triplet_subset = self.sample_triplets(triplet_subset)
        triplet_subset = triplet_subset.ravel()
        X = self.X[triplet_subset]
        X = rearrange(X, "(n k) h w c -> n k h w c", n=X.shape[0] // 3)
        X = jax.device_put(X)
        y = jax.device_put(y)
        return (X, y)

    def main_batch_balancing(self) -> Tuple[Array, Array]:
        """Sample classes uniformly for each randomly sampled mini-batch."""
        for _ in range(self.num_batches):
            main_batch = self.sample_main_batch()
            yield main_batch

    def mtl_batch_balancing(
        self,
    ) -> Tuple[Tuple[Array, Array], Tuple[Array, Array]]:
        """Simultaneously sample odd-one-out triplet and main multi-class task mini-batches."""
        for _ in range(self.num_batches):
            ooo_batch = self.sample_ooo_batch()
            main_batch = self.sample_main_batch()
            yield main_batch, ooo_batch

    def __iter__(self) -> Iterator:
        if self.model_config.task == "mtl":
            return iter(self.mtl_batch_balancing())
        else:
            if self.data_config.sampling == "standard":
                if self.train:
                    # randomly permute the order of samples in the data (i.e., for each epoch shuffle the data)
                    random_order = np.random.permutation(np.arange(len(self.dataset)))
                return iter(self.stepping(random_order))
            else:
                return iter(self.main_batch_balancing())

    def __len__(self) -> int:
        return self.num_batches
