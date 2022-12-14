#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataLoader"]

import copy
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Iterator, List, Tuple, Union

import dm_pix as pix
import haiku as hk
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

    def __post_init__(self) -> None:
        self.cpu_devices = jax.devices("cpu")
        num_gpus = 2
        self.device_num = random.choices(range(num_gpus))[0]
        self.X = np.asarray(self.data[0])
        self.y = copy.deepcopy(self.data[1])
        self.y = jax.device_put(self.y)

        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.rng_seq = hk.PRNGSequence(self.seed)

        if self.data_config.name.endswith("mnist"):
            self.X = np.expand_dims(self.X, axis=-1)

        self.num_classes = self.y.shape[-1]
        self.y_prime = jnp.nonzero(self.y)[-1]
        self.oko_classes = np.unique(self.y_prime)

        if self.train:
            self.set_card = self.data_config.k + 2
            self.num_batches = math.ceil(
                self.data_config.num_sets / self.data_config.oko_batch_size
            )
        else:
            self.num_batches = math.ceil(
                self.X.shape[0] / self.data_config.main_batch_size
            )
            self.remainder = self.X.shape[0] % self.data_config.main_batch_size

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
        def sample_member(
            classes: Array, num_set_classes: int, q: float, key: Array
        ) -> Array:
            return jax.random.choice(
                key, classes, shape=(num_set_classes,), replace=False, p=q
            )

        @partial(jax.jit, static_argnames=["seed"])
        def sample_members(seed: int, q=None) -> Array:
            """Sample pairs of objects from the same class."""
            key = jax.random.PRNGKey(seed)
            keys = jax.random.split(key, num=self.data_config.oko_batch_size)
            return vmap(partial(self.sample_member, q))(keys)

        @jaxtyped
        @typechecker
        def sample_set_instances(
            y_prime: Int32[Array, "n"], set: Int32[np.ndarray, "k"]
        ) -> List[np.int32]:
            """Uniformly sample instances/indices for the two classes in a set without replacement."""
            instances = []
            for cls in np.unique(set):
                num_examples = np.count_nonzero(set == cls)
                rnd_sample = np.random.choice(
                    np.where(y_prime == cls)[0],
                    size=num_examples,
                    replace=False,  # sample instances uniformly (p = None) without replacement
                    p=None,
                ).astype(np.int32)
                instances.extend(rnd_sample)
            return instances

        @partial(jax.jit, static_argnames=["num_cls", "set_card", "k"])
        def make_bimodal_targets(
            num_cls: int,
            set_card: int,
            k: int,
            pair_classes: Int32[np.ndarray, "#batch"],
            oko_classes: Int32[np.ndarray, "#batch"],
        ) -> Float32[Array, "#batch num_cls"]:
            y_p = jax.nn.one_hot(x=pair_classes, num_classes=num_cls) * (set_card - k)
            y_o = jax.nn.one_hot(x=oko_classes, num_classes=num_cls)
            y = (y_p + y_o) / set_card
            return y

        @partial(jax.jit, static_argnames=["num_cls", "set_card", "k"])
        def make_multimodal_targets(
            num_cls: int,
            set_card: int,
            k: int,
            pair_classes: Int32[np.ndarray, "#batch"],
            odd_classes: Int32[np.ndarray, "#batch k"],
        ) -> Float32[Array, "#batch num_cls"]:
            y = jax.nn.one_hot(x=pair_classes, num_classes=num_cls) * (set_card - k)
            for classes in odd_classes.T:
                y += jax.nn.one_hot(x=classes, num_classes=num_cls)
            y /= set_card
            return y

        # jit or partially initialize functions for computational efficiency
        if self.train:
            self.sample_member = partial(
                sample_member, self.oko_classes, self.set_card - 1
            )
            self.sample_members = sample_members
            self.sample_set_instances = partial(sample_set_instances, self.y_prime)
            if self.data_config.targets == "soft":
                if self.data_config.k == 1:
                    self._make_bimodal_targets = partial(
                        make_bimodal_targets,
                        self.num_classes,
                        self.set_card,
                        self.data_config.k,
                    )
                else:
                    self._make_multimodal_targets = partial(
                        make_multimodal_targets,
                        self.num_classes,
                        self.set_card,
                        self.data_config.k,
                    )

        self._make_augmentations()

    def _make_augmentations(self) -> None:
        self.flip_left_right = jax.jit(pix.random_flip_left_right)
        self.augmentations = [self.flip_left_right]

        if self.data_config.name.lower() == "fashionmnist":
            self.flip_up_down = jax.jit(pix.random_flip_up_down)
            self.augmentations.append(self.flip_up_down)

        if self.data_config.name.lower().startswith("cifar"):
            self.rnd_crop = jax.jit(pix.random_crop)
            self.augmentations.append(self.rnd_crop)
            

    @jaxtyped
    @typechecker
    def apply_augmentations(
        self, batch: UInt8orFP32[Array, "#batchk h w c"]
    ) -> UInt8orFP32[Array, "#batchk h w c"]:
        for i, augmentation in enumerate(self.augmentations):
            if (
                self.data_config.name.startswith("cifar")
                and i == len(self.augmentations) - 1
            ):
                batch = augmentation(key=next(self.rng_seq), image=batch, crop_size=[batch.shape[1]])
            else:
                batch = augmentation(key=next(self.rng_seq), image=batch)
        return batch

    @staticmethod
    @jaxtyped
    @typechecker
    def make_sets(
        members: Int32[Array, "#batch _"],
        pair_classes: Int32[np.ndarray, "#batch"],
    ) -> Int32[np.ndarray, "#batch card"]:
        """Make b sets with k+2 members, where k denotes the number of odd classes."""
        # return np.c_[pair_classes, members, pair_classes]
        return np.c_[members, pair_classes]

    @jaxtyped
    @typechecker
    def get_odd_classes(
        self,
        sets: Int32[np.ndarray, "#batch _"],
        pair_classes: Int32[np.ndarray, "#batch"],
    ) -> Int32[np.ndarray, "#batch k"]:
        """Find the k odd classes per set."""
        if self.data_config["k"] == 1:
            # a single odd class in a set
            odd_classes = np.array(
                [
                    set[np.where(set != sim_cls)[0][0]]
                    for set, sim_cls in zip(sets, pair_classes)
                ]
            )
        else:
            # multiple odd classes in a set
            odd_classes = np.array(
                [
                    set[np.where(set != sim_cls)[0]]
                    for set, sim_cls in zip(sets, pair_classes)
                ]
            )
        return odd_classes

    @staticmethod
    @jaxtyped
    @typechecker
    def choose_pair_classes(
        members: Int32[Array, "#batch _"]
    ) -> Int32[np.ndarray, "#batch"]:
        """Randomly choose a pair class from all k+1 classes in a set with k+1 members (each member represents an instance from a class)."""
        return np.apply_along_axis(np.random.choice, axis=1, arr=members)

    # @jaxtyped
    # @typechecker
    def create_sets(
        self, members: Int32[Array, "#batch _"]
    ) -> Union[
        Tuple[
            Int32[np.ndarray, "#batch _"],
            Int32[np.ndarray, "#batch 2"],
        ],
        Tuple[
            Int32[np.ndarray, "#batch _"],
            Int32[np.ndarray, "#batch 2"],
            Int32[np.ndarray, "#batch k"],
        ],
    ]:
        pair_classes = self.choose_pair_classes(members)
        if self.data_config.k > 0:
            # odd-k-out learning
            sets = self.make_sets(
                members=members,
                pair_classes=pair_classes,
            )
            sets = np.apply_along_axis(np.random.permutation, axis=1, arr=sets)
            if self.data_config["targets"] == "soft":
                odd_classes = self.get_odd_classes(sets, pair_classes)
                return sets, pair_classes, odd_classes
        else:
            # pair learning (i.e., set cardinality = 2)
            sets = np.c_[pair_classes, pair_classes]
            sets = np.apply_along_axis(np.random.permutation, axis=1, arr=sets)
        return sets, pair_classes

    @jaxtyped
    @typechecker
    def sample_batch_instances(
        self, sets: Int32[np.ndarray, "#batch k"]
    ) -> Int32[np.ndarray, "#batch k"]:
        """Sample unique instances/indices from the classes in each set."""
        return np.apply_along_axis(self.sample_set_instances, arr=sets, axis=1)

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
                subset = list(
                    range(
                        i * self.data_config.main_batch_size,
                        i * self.data_config.main_batch_size + self.remainder,
                    )
                )
            else:
                subset = list(
                    range(
                        i * self.data_config.main_batch_size,
                        (i + 1) * self.data_config.main_batch_size,
                    )
                )
            X = jax.device_put(self.X[np.asarray(subset)])
            y = self.y[np.asarray(subset)]
            yield (X, y)

    @jaxtyped
    @typechecker
    def sample_oko_batch(
        self, q=None
    ) -> Tuple[UInt8orFP32[Array, "#batchk h w c"], Float32[Array, "#batch num_cls"]]:
        """Uniformly sample odd-one-out triplet task mini-batches."""
        seed = np.random.randint(low=0, high=1e9, size=1)[0]
        set_members = self.sample_members(seed, q=q)

        if self.data_config["targets"] == "soft":
            # create soft targets that reflect the true probability distribution of classes in a set
            sets, pair_classes, odd_classes = self.create_sets(set_members)
            if self.data_config.k == 1:
                y = self._make_bimodal_targets(pair_classes, odd_classes)
            else:
                y = self._make_multimodal_targets(pair_classes, odd_classes)
        else:
            # create "hard" targets with a point mass at the pair class
            sets, pair_classes = self.create_sets(set_members)
            y = jax.nn.one_hot(x=pair_classes, num_classes=self.num_classes)

        batch_sets = self.sample_batch_instances(sets)
        batch_sets = batch_sets.ravel()
        X = jax.device_put(self.X[batch_sets])
        if self.data_config.apply_augmentations:
            X = self.apply_augmentations(X)
        return (X, y)

    def smoothing(self) -> Array:
        @jax.jit
        def softmax(p: Array, beta: float) -> Array:
            return jnp.exp(p / beta) / (jnp.exp(p / beta).sum())

        return partial(softmax, self.p)(self.temperature)

    @jaxtyped
    @typechecker
    def oko_batch_balancing(
        self,
    ) -> Iterator[
        Tuple[UInt8orFP32[Array, "#batchk h w c"], Float32[Array, "#batch num_cls"]]
    ]:
        """Simultaneously sample odd-one-out triplet and main multi-class task mini-batches."""
        q = self.smoothing() if self.data_config.sampling == "dynamic" else None
        for _ in range(self.num_batches):
            oko_batch = self.sample_oko_batch(q)
            yield oko_batch
        if self.data_config.sampling == "dynamic":
            self.temperature += 0.1

    def __iter__(self) -> Iterator:
        if self.train:
            return iter(self.oko_batch_balancing())
        return iter(self.stepping())

    def __len__(self) -> int:
        return self.num_batches
