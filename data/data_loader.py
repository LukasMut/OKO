#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["OKOLoader"]

import copy
import math
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import dm_pix as pix
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax.tree_util import register_pytree_node_class
from jaxtyping import AbstractDtype, Array, Float32, Float64, Int32, jaxtyped
from ml_collections import config_dict
from typeguard import typechecked as typechecker

from data.augmentations import RandomCrop

FrozenDict = config_dict.FrozenConfigDict
PRNGSequence = Any


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


class UInt8orFP64(AbstractDtype):
    dtypes = ["uint8", "float64"]


class Int32or64(AbstractDtype):
    dtypes = ["int8", "int64"]


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class SetMaker:
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
        return np.apply_along_axis(
            np.random.permutation, axis=1, arr=np.c_[members, pair_classes]
        )

    @jaxtyped
    @typechecker
    def get_odd_classes(
        self, set: Int32[Array, "set_card"], pair_cls: Int32[Array, ""]
    ) -> Int32[Array, "k"]:
        """Each set has k odd classes."""
        return set[jnp.where(set != pair_cls, size=self.num_odds)[0]]

    @jaxtyped
    @typechecker
    def vget_odd_classes(
        self,
        sets: Int32[np.ndarray, "#batch set_card"],
        pair_classes: Int32[np.ndarray, "#batch"],
    ) -> Union[Int32[Array, "#batch"], Int32[Array, "#batch k"]]:
        """Get the k odd classes for a batch of sets."""
        odd_classes = vmap(self.get_odd_classes)(sets, pair_classes)
        if self.num_odds == 1:
            # if there's a single odd class in each set, flatten array
            odd_classes = odd_classes.ravel()
        return odd_classes

    @staticmethod
    @jaxtyped
    @typechecker
    def choose_pair_classes(
        members: Int32[Array, "#batch _"]
    ) -> Int32[np.ndarray, "#batch"]:
        """Randomly choose a pair class from all k+1 classes in a set with k+1 members (each member represents an instance from a class)."""
        return np.apply_along_axis(np.random.choice, axis=1, arr=members)

    @jaxtyped
    @typechecker
    def _make_sets(
        self, members: Int32[Array, "#batch _"]
    ) -> Union[
        Tuple[
            Int32[np.ndarray, "#batch set_card"],
            Int32[np.ndarray, "#batch"],
            Int32[Array, "#batch"],
        ],
        Tuple[
            Int32[np.ndarray, "#batch set_card"],
            Int32[np.ndarray, "#batch"],
            Int32[Array, "#batch k"],
        ],
    ]:
        pair_classes = members[:, 0]
        sets = self.make_sets(
            members=members,
            pair_classes=pair_classes,
        )
        odd_classes = self.vget_odd_classes(sets, pair_classes)
        return sets, pair_classes, odd_classes


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class TargetMaker:
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

    @jaxtyped
    @typechecker
    def make_bimodal_targets(
        self,
        pair_classes: Int32[Array, "#batch"],
        odd_classes: Int32[Array, "#batch"],
    ) -> Float32[Array, "#batch num_cls"]:
        y_p = jax.nn.one_hot(x=pair_classes, num_classes=self.num_cls) * (
            self.set_card - self.num_odds
        )
        y_o = jax.nn.one_hot(x=odd_classes, num_classes=self.num_cls)
        y = (y_p + y_o) / self.set_card
        return y

    @jaxtyped
    @typechecker
    def make_multimodal_targets(
        self,
        pair_classes: Int32[Array, "#batch"],
        odd_classes: Int32[Array, "#batch k"],
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
        pair_classes: Int32[Array, "#batch"],
        odd_classes: Union[Int32[Array, "#batch"], Int32[Array, "#batch k"]],
    ) -> Float32[Array, "#batch num_cls"]:
        if self.num_odds == 1:
            y = self.make_bimodal_targets(pair_classes, odd_classes)
        else:
            y = self.make_multimodal_targets(pair_classes, odd_classes)
        return y


@register_pytree_node_class
@dataclass(init=True, repr=True, frozen=True)
class Sampler:
    classes: Int32or64[Array, "num_cls"]
    batch_size: int
    num_set_classes: int
    random_numbers: Iterator

    def tree_flatten(self) -> Tuple[Tuple[Int32[Array, "num_cls"]], Dict[str, Any]]:
        children = (self.classes,)
        aux_data = {
            "batch_size": self.batch_size,
            "num_set_classes": self.num_set_classes,
            "random_numbers": self.random_numbers,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def sample_member(self, key: Array) -> Array:
        return jax.random.choice(
            key, self.classes, shape=(self.num_set_classes,), replace=False, p=None
        )

    def get_key(self) -> Array:
        return jax.random.PRNGKey(next(self.random_numbers))

    @jax.jit
    def sample_members(self) -> Array:
        """Sample pairs of objects from the same class."""
        keys = jax.random.split(self.get_key(), num=self.batch_size)
        return vmap(self.sample_member)(keys)


@dataclass(init=True, repr=True)
class OKOLoader:
    data: Tuple[Array, Array]
    data_config: FrozenDict
    model_config: FrozenDict
    seed: int
    train: bool = True

    def __post_init__(self) -> None:
        self.X = np.array(self.data[0])
        self.y = copy.deepcopy(self.data[1])

        # seed random number generator
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.rng_seq = hk.PRNGSequence(self.seed)

        self.num_classes = self.y.shape[-1]

        if self.train and self.data_config.label_noise:
            self.y = self.swap_labels(self.y)

        self.y_prime = jnp.nonzero(self.y)[-1]
        self.classes = jnp.unique(self.y_prime)

        if self.train:
            self.set_card = self.data_config.k + 2
            self.num_batches = math.ceil(
                self.data_config.num_sets / self.data_config.oko_batch_size
            )
            max_num = math.ceil(self.num_batches * self.data_config.epochs)
            self.sampler = Sampler(
                self.classes,
                self.data_config.oko_batch_size,
                self.set_card - 1,
                iter(np.random.permutation(max_num)),
            )
            self.set_maker = SetMaker(self.data_config.k, self.data_config.targets)
            if self.data_config.targets == "soft":
                assert (
                    self.data_config.k > 0
                ), "\nIf you want to use soft labels, there must at least be one odd class.\n"
                self.target_maker = TargetMaker(
                    self.num_classes, self.data_config.k, self.set_card
                )

            self._create_functions()
            self._make_augmentations(max_num=max_num)
        else:
            self.num_batches = math.ceil(
                self.X.shape[0] / self.data_config.main_batch_size
            )
            self.remainder = self.X.shape[0] % self.data_config.main_batch_size

    def _create_functions(self) -> None:
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

        # partially initialize functions for more computational efficiency
        if self.train:
            self.sample_set_instances = partial(sample_set_instances, self.y_prime)

    def _make_augmentations(self, max_num: Optional[int] = None) -> None:
        if self.data_config.name.lower() == "mnist":
            self.flip_left_right = jax.jit(pix.random_flip_left_right)
            self.augmentations = [self.flip_left_right]

        elif self.data_config.name.lower() == "fashion_mnist":
            self.flip_left_right = jax.jit(pix.random_flip_left_right)
            self.augmentations = [self.flip_left_right]

        elif self.data_config.name.lower().startswith("cifar"):
            self.rnd_crop = RandomCrop(
                crop_size=self.X.shape[1:],
                random_numbers=iter(np.random.permutation(max_num)),
            )
            self.flip_left_right = jax.jit(pix.random_flip_left_right)
            self.augmentations = [self.rnd_crop, self.flip_left_right]

    # @jaxtyped
    # @typechecker
    def apply_augmentations(
        self,
        batch: Union[
            UInt8orFP64[np.ndarray, "#batchk h w c"],
            UInt8orFP32[Array, "#batchk h w c"],
        ],
    ) -> UInt8orFP32[Array, "#batchk h w c"]:
        subbatch_i = batch[::2]
        subbatch_j = batch[1::2]
        for i, augmentation in enumerate(self.augmentations):
            if self.data_config.name.startswith("cifar") and i == 0:
                # augmented_batch = vmap(
                #    lambda x: jax.image.resize(
                #        x,
                #        shape=(x.shape[0] + 8, x.shape[0] + 8, x.shape[-1]),
                #        method="bilinear",
                #        antialias=True,
                #    )
                # )(subbatch_j)
                augmented_batch = vmap(lambda x: jnp.pad(x, pad_width=4, mode="edge"))(
                    subbatch_j
                )
                subbatch_j = augmentation.apply(augmented_batch)
            else:
                subbatch_j = vmap(
                    lambda x: augmentation(key=next(self.rng_seq), image=x)
                )(subbatch_j)
        batch = jnp.array(list(zip(subbatch_i, subbatch_j)))
        batch = batch.reshape(-1, *batch.shape[2:])
        if subbatch_i.shape[0] > subbatch_j.shape[0]:
            batch = jnp.concatenate((batch, subbatch_i[-1][None, ...]), axis=0)
        return batch

    # @jaxtyped
    # @typechecker
    def _normalize(
        self,
        batch: Union[
            UInt8orFP32[Array, "#batchk h w c"],
            UInt8orFP64[np.ndarray, "#batchk h w c"],
        ],
    ) -> Union[
        UInt8orFP32[Array, "#batchk h w c"],
        UInt8orFP64[np.ndarray, "#batchk h w c"],
    ]:
        batch = batch / self.data_config.max_pixel_value
        batch -= self.data_config.means
        batch /= self.data_config.stds
        return batch

    @staticmethod
    def random_permute(x: Float32[Array, "num_cls"]) -> Float64[np.ndarray, "num_cls"]:
        swap = np.random.choice(a=[0, 1], size=1, p=[0.8, 0.2]).item()
        if swap:
            x = np.random.permutation(x)
        return x

    def swap_labels(
        self, y: Float32[Array, "n num_cls"]
    ) -> Float64[np.ndarray, "n num_cls"]:
        return np.apply_along_axis(self.random_permute, axis=1, arr=y)

    @jaxtyped
    @typechecker
    def sample_batch_instances(
        self, sets: Int32[np.ndarray, "#batch set_card"]
    ) -> Int32[np.ndarray, "#batch set_card"]:
        """Sample unique instances/indices from the classes in each set."""
        return np.apply_along_axis(self.sample_set_instances, arr=sets, axis=1)

    @jaxtyped
    @typechecker
    def stepping(
        self,
    ) -> Iterator[
        Tuple[
            Union[
                UInt8orFP32[Array, "#batch h w c"],
                UInt8orFP32[np.ndarray, "#batchk h w c"],
                UInt8orFP64[np.ndarray, "#batchk h w c"],
            ],
            Float32[Array, "#batch num_cls"],
        ]
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
            X = self.X[np.asarray(subset)]
            y = self.y[np.asarray(subset)]
            yield (X, y)

    # @jaxtyped
    # @typechecker
    def sample_oko_batch(
        self,
    ) -> Tuple[
        Union[
            UInt8orFP32[Array, "#batchk h w c"],
            UInt8orFP32[np.ndarray, "#batchk h w c"],
        ],
        Float32[Array, "#batch num_cls"],
    ]:
        """Uniformly sample odd-one-out triplet task mini-batches."""
        set_members = self.sampler.sample_members()
        sets, pair_classes, odd_classes = self.set_maker._make_sets(set_members)
        # create "hard" targets with a point mass at the pair class
        y_p = jax.nn.one_hot(x=pair_classes, num_classes=self.num_classes)
        # create "hard" targets with a point mass at the odd class
        y_n = jax.nn.one_hot(x=odd_classes, num_classes=self.num_classes)
        batch_sets = self.sample_batch_instances(sets)
        X = self.X[batch_sets.ravel()]
        if self.data_config.apply_augmentations:
            X = self.apply_augmentations(X)
        if self.data_config.is_rgb_dataset:
            X = self._normalize(X)
        return (X, (y_p, y_n))

    # @jaxtyped
    # @typechecker
    def oko_batch_balancing(
        self,
    ) -> Iterator[
        Tuple[
            Union[
                UInt8orFP32[Array, "#batchk h w c"],
                UInt8orFP32[np.ndarray, "#batchk h w c"],
            ],
            Float32[Array, "#batch num_cls"],
        ]
    ]:
        """Simultaneously sample odd-one-out triplet and main multi-class task mini-batches."""
        for _ in range(self.num_batches):
            yield self.sample_oko_batch()

    def __iter__(self) -> Iterator:
        if self.train:
            return iter(self.oko_batch_balancing())
        return iter(self.stepping())

    def __len__(self) -> int:
        return self.num_batches
