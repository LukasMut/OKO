#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["OKOLoader"]

import copy
import math
import random
from collections import Counter
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterator, List, Tuple, Union

import dm_pix as pix
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jaxtyping import AbstractDtype, Array, Float32, Int32, jaxtyped
from ml_collections import config_dict
from typeguard import typechecked as typechecker

FrozenDict = config_dict.FrozenConfigDict


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class MakeSets:
    num_odds: int
    target_type: str

    def tree_flatten(self) -> Tuple[tuple, Dict[str, Any]]:
        children = ()
        aux_data = {"num_odds": self.num_odds, "targets": self.target_type}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    @staticmethod
    @jaxtyped
    @typechecker
    def make_sets(
        members: Int32[Array, "#batch _"],
        pair_classes: Int32[np.ndarray, "#batch"],
    ) -> Int32[np.ndarray, "#batch set_card"]:
        """Make b sets with k+2 members (i.e., set_card = k+2), where k denotes the number of odd classes in the set."""
        return np.c_[members, pair_classes]

    @jaxtyped
    @typechecker
    def get_odd_classes(
        self,
        sets: Int32[np.ndarray, "#batch _"],
        pair_classes: Int32[np.ndarray, "#batch"],
    ) -> Union[Int32[np.ndarray, "#batch k"], Int32[np.ndarray, "#batch"]]:
        """Find the k odd classes per set."""
        if self.num_odds:
            # there's a single odd class in a set
            odd_classes = np.array(
                [
                    set[np.where(set != sim_cls)[0][0]]
                    for set, sim_cls in zip(sets, pair_classes)
                ]
            )
        else:
            # there are multiple odd classes in a set
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
    def create(
        self, members: Int32[Array, "#batch _"]
    ) -> Union[
        Tuple[
            Int32[np.ndarray, "#batch _"],
            Int32[np.ndarray, "#batch 2"],
            Int32[np.ndarray, "#batch"],
        ],
        Tuple[
            Int32[np.ndarray, "#batch _"],
            Int32[np.ndarray, "#batch 2"],
            Int32[np.ndarray, "#batch k"],
        ],
        Tuple[
            Int32[np.ndarray, "#batch _"],
            Int32[np.ndarray, "#batch 2"],
        ],
    ]:
        pair_classes = self.choose_pair_classes(members)
        if self.num_odds > 0:
            # odd-k-out learning
            sets = self.make_sets(
                members=members,
                pair_classes=pair_classes,
            )
            sets = np.apply_along_axis(np.random.permutation, axis=1, arr=sets)
            if self.target_type == "soft":
                odd_classes = self.get_odd_classes(sets, pair_classes)
                return sets, pair_classes, odd_classes
        else:
            # pair learning (i.e., set cardinality = 2)
            sets = np.c_[pair_classes, pair_classes]
            sets = np.apply_along_axis(np.random.permutation, axis=1, arr=sets)
        return sets, pair_classes


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class MakeTargets:
    num_cls: int
    num_odds: int
    set_card: int

    def tree_flatten(self) -> Tuple[tuple, Dict[str, Any]]:
        children = ()
        aux_data = {
            "num_cls": self.num_cls,
            "num_odds": self.num_odds,
            "set_card": self.set_card,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def make_bimodal_targets(
        self,
        pair_classes: Int32[np.ndarray, "#batch"],
        oko_classes: Int32[np.ndarray, "#batch"],
    ) -> Float32[Array, "#batch num_cls"]:
        y_p = jax.nn.one_hot(x=pair_classes, num_classes=self.num_cls) * (
            self.set_card - self.num_odds
        )
        y_o = jax.nn.one_hot(x=oko_classes, num_classes=self.num_cls)
        y = (y_p + y_o) / self.set_card
        return y

    def make_multimodal_targets(
        self,
        pair_classes: Int32[np.ndarray, "#batch"],
        odd_classes: Int32[np.ndarray, "#batch k"],
    ) -> Float32[Array, "#batch num_cls"]:
        y = jax.nn.one_hot(x=pair_classes, num_classes=self.num_cls) * (
            self.set_card - self.num_odds
        )
        for classes in odd_classes.T:
            y += jax.nn.one_hot(x=classes, num_classes=self.num_cls)
        y /= self.set_card
        return y

    @jax.jit
    def _make_targets(
        self,
        pair_classes: Int32[np.ndarray, "#batch"],
        odd_classes: Int32[np.ndarray, "#batch k"],
    ) -> Float32[Array, "#batch num_cls"]:
        if self.num_odds == 1:
            y = self.make_bimodal_targets(pair_classes, odd_classes)
        else:
            y = self.make_multimodal_targets(pair_classes, odd_classes)
        return y


@dataclass(init=True, repr=True)
class OKOLoader:
    data: Tuple[Array, Array]
    data_config: FrozenDict
    model_config: FrozenDict
    seed: int
    train: bool = True

    def __post_init__(self) -> None:
        self.X = np.asarray(self.data[0])
        self.y = jax.device_put(copy.deepcopy(self.data[1]))

        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.rng_seq = hk.PRNGSequence(self.seed)

        self.num_classes = self.y.shape[-1]
        self.y_prime = jnp.nonzero(self.y)[-1]
        self.classes = np.unique(self.y_prime)

        if self.train:
            self.set_card = self.data_config.k + 2
            self.num_batches = math.ceil(
                self.data_config.num_sets / self.data_config.oko_batch_size
            )
            self.set_maker = MakeSets(self.data_config.k, self.data_config.targets)
            if self.data_config.targets == "soft":
                self.target_maker = MakeTargets(
                    self.num_classes, self.set_card, self.data_config.k
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
            y_prime: Int32[Array, "n"], set: Int32[np.ndarray, "set_card"]
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

        # jit or partially initialize functions for computational efficiency
        if self.train:
            self.sample_member = partial(sample_member, self.classes, self.set_card - 1)
            self.sample_members = sample_members
            self.sample_set_instances = partial(sample_set_instances, self.y_prime)

        self._make_augmentations()

    def _make_augmentations(self) -> None:
        if self.data_config.name.lower() == "mnist":
            self.flip_left_right = jax.jit(pix.random_flip_left_right)
            self.augmentations = [self.flip_left_right]

        elif self.data_config.name.lower() == "fashionmnist":
            self.flip_left_right = jax.jit(pix.random_flip_left_right)
            self.flip_up_down = jax.jit(pix.random_flip_up_down)
            self.augmentations = [self.flip_left_right, self.flip_up_down]

        elif self.data_config.name.lower().startswith("cifar"):
            self.rnd_crop = pix.random_crop
            self.flip_left_right = jax.jit(pix.random_flip_left_right)
            self.augmentations = [self.rnd_crop, self.flip_left_right]

    @jaxtyped
    @typechecker
    def apply_augmentations(
        self, batch: UInt8orFP32[Array, "#batchk h w c"]
    ) -> UInt8orFP32[Array, "#batchk h w c"]:
        for i, augmentation in enumerate(self.augmentations):
            if self.data_config.name.startswith("cifar") and i == 0:
                batch = augmentation(
                    key=next(self.rng_seq), image=batch, crop_sizes=batch.shape
                )
            else:
                batch = augmentation(key=next(self.rng_seq), image=batch)
        return batch

    @jaxtyped
    @typechecker
    def _normalize(
        self, batch: UInt8orFP32[Array, "#batchk h w c"]
    ) -> UInt8orFP32[Array, "#batchk h w c"]:
        batch = batch / self.data_config.max_pixel_value
        batch -= self.data_config.means
        batch /= self.data_config.stds
        return batch

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
        if self.data_config.targets == "soft":
            # create soft targets that reflect the true probability distribution of classes in a set
            sets, pair_classes, odd_classes = self.set_maker.create(set_members)
            y = self.target_maker._make_targets(pair_classes, odd_classes)
        else:
            # create "hard" targets with a point mass at the pair class
            sets, pair_classes = self.set_maker.create(set_members)
            y = jax.nn.one_hot(x=pair_classes, num_classes=self.num_classes)
        batch_sets = self.sample_batch_instances(sets)
        batch_sets = batch_sets.ravel()
        X = jax.device_put(self.X[batch_sets])
        if self.data_config.apply_augmentations:
            X = self.apply_augmentations(X)
        if self.data_config.is_rgb_dataset:
            X = self._normalize(X)
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
