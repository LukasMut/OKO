#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataLoader"]

from cmath import exp
import copy
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import partial
from signal import raise_signal
from typing import Iterator, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jaxtyping import AbstractDtype, Array, Float32, Int32, jaxtyped
from ml_collections import config_dict
from typeguard import typechecked as typechecker

FrozenDict = config_dict.FrozenConfigDict


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


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
        num_gpus = 2
        self.device_num = random.choices(range(num_gpus))[0]
        self.X = self.data[0]
        self.y = copy.deepcopy(self.data[1])

        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.data_config.name.endswith("mnist"):
            self.X = jnp.expand_dims(self.X, axis=-1)

        self.num_classes = self.y.shape[-1]
        self.y_prime = jnp.nonzero(self.y)[-1]
        self.ooo_classes = np.unique(self.y_prime)

        if self.train:
            self.k = 3
            self.num_batches = math.ceil(
                self.data_config.max_triplets / self.data_config.ooo_batch_size
            )
        else:
            
            self.dataset = list(zip(self.X, self.y))
            self.num_batches = math.ceil(
                len(self.dataset) / self.data_config.main_batch_size
            )
            self.remainder = len(self.dataset) % self.data_config.main_batch_size

        if self.data_config.sampling == "dynamic":
            self.y_flat = np.nonzero(self.y)[1]
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

        @jaxtyped
        @typechecker
        def sample_triplet(
            y_prime: Int32[Array, "n"], triplet: Int32[np.ndarray, "k"]
        ) -> List[np.int32]:
            """Uniformly sample instances/indices for the two classes in a triplet without replacement."""
            instances = []
            for cls in np.unique(triplet):
                num_examples = np.count_nonzero(triplet == cls)
                rnd_sample = np.random.choice(
                    np.where(y_prime == cls)[0],
                    size=num_examples,
                    replace=False,  # sample instances uniformly without replacement
                    p=None,
                ).astype(np.int32)
                instances.extend(rnd_sample)
            return instances

        @jaxtyped
        @typechecker
        def unzip_pairs(
            dataset: List[
                Tuple[UInt8orFP32[Array, "h w c"], Float32[np.ndarray, "num_cls"]]
            ],
            subset: range,
        ) -> Tuple[UInt8orFP32[Array, "#batch h w c"], Float32[Array, "#batch num_cls"]]:
            """Create tuples of data pairs (X, y)."""
            X, y = zip(*[dataset[i] for i in subset])
            X = jnp.stack(X, axis=0)
            y = jnp.stack(y, axis=0)
            return (X, y)

        # jit or partially initialize functions for computational efficiency
        if self.train:
            self.sample_double = partial(sample_double, self.ooo_classes)
            self.sample_doubles = sample_doubles
            self.sample_triplet = partial(sample_triplet, self.y_prime)
        else:
            self.unzip_pairs = partial(unzip_pairs, self.dataset)

    @staticmethod
    @jaxtyped
    @typechecker
    def make_tuples(
        doubles: Int32[Array, "#batch 2"],
        pair_classes: Int32[np.ndarray, "#batch"],
        k: int,
    ) -> Int32[np.ndarray, "#batch k"]:
        """Make b ordered tuples of k-1 "pair" class instances and one odd-one-out."""
        if k == 3:
            tuples = np.c_[doubles, pair_classes]
        else:
            tuples = np.c_[doubles, pair_classes, pair_classes]
        return tuples

    @jaxtyped
    @typechecker
    def expand(
        self, doubles: Int32[Array, "#batch 2"]
    ) -> Tuple[Int32[np.ndarray, "#batch k"], Int32[np.ndarray, "#batch"]]:
        pair_classes = np.apply_along_axis(np.random.choice, axis=1, arr=doubles)
        tuples = self.make_tuples(doubles=doubles, pair_classes=pair_classes, k=self.k)
        tuples = np.apply_along_axis(np.random.permutation, axis=1, arr=tuples)
        """
        ooo_classes = np.array(
            [
                triplet[np.where(triplet != sim_cls)[0][0]]
                for triplet, sim_cls in zip(triplets, pair_classes)
            ]
        )
        return tuples, ooo_classes

        """
        return tuples, pair_classes

    @jaxtyped
    @typechecker
    def sample_triplets(
        self, triplets: Int32[np.ndarray, "#batch k"]
    ) -> Int32[np.ndarray, "#batch k"]:
        """Sample instances/indices from the corresponding classes."""
        return np.apply_along_axis(self.sample_triplet, arr=triplets, axis=1)

    @jaxtyped
    @typechecker
    def stepping(
        self,
    ) -> Iterator[
        Tuple[UInt8orFP32[Array, "#batch h w c"], Float32[Array, "#batch num_cls"]]
    ]:
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

    @jaxtyped
    @typechecker
    def sample_ooo_batch(
        self, q=None
    ) -> Tuple[UInt8orFP32[Array, "#batchk h w c"], Float32[Array, "#batch num_cls"]]:
        """Uniformly sample odd-one-out triplet task mini-batches."""
        seed = np.random.randint(low=0, high=1e9, size=1)[0]
        doubles_subset = self.sample_doubles(seed, q=q)
        triplet_subset, pair_classes = self.expand(doubles_subset)
        y = jax.nn.one_hot(x=pair_classes, num_classes=self.num_classes)
        triplet_subset = self.sample_triplets(triplet_subset)
        triplet_subset = triplet_subset.ravel()
        X = self.X[triplet_subset]
        X = jax.device_put(X)
        return (X, y)

    def smoothing(self) -> Array:
        @jax.jit
        def softmax(p: Array, beta: float) -> Array:
            return jnp.exp(p / beta) / (jnp.exp(p / beta).sum())

        return partial(softmax, self.p)(self.temperature)

    @jaxtyped
    @typechecker
    def ooo_batch_balancing(
        self,
    ) -> Iterator[
        Tuple[UInt8orFP32[Array, "#batchk h w c"], Float32[Array, "#batch num_cls"]]
    ]:
        """Simultaneously sample odd-one-out triplet and main multi-class task mini-batches."""
        q = self.smoothing() if self.data_config.sampling == "dynamic" else None
        for _ in range(self.num_batches):
            ooo_batch = self.sample_ooo_batch(q)
            yield ooo_batch
        if self.data_config.sampling == "dynamic":
            self.temperature += 0.1

    def __iter__(self) -> Iterator:
        if self.train:
            return iter(self.ooo_batch_balancing())
        return iter(self.stepping())

    def __len__(self) -> int:
        return self.num_batches
