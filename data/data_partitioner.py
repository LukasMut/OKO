#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataPartitioner"]

import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import AbstractDtype, Array, Float32, Int32, jaxtyped
from typeguard import typechecked as typechecker

np_array = np.ndarray
jnp_array = jnp.ndarray


RGB_DATASETS = ["cifar10", "cifar100", "imagenet"]


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


@dataclass(init=True, repr=True)
class DataPartitioner:
    dataset: str
    data_path: str
    n_samples: int
    seed: int
    probability_mass: float
    min_samples: int = None
    train: bool = True
    train_frac: float = 0.85

    def __post_init__(self) -> None:
        # seed rng
        random.seed(self.seed)
        np.random.seed(self.seed)
        assert isinstance(
            self.min_samples, int
        ), "\nMinimum number of samples per class must be defined.\n"
        self.max_pixel_value = np.array(255.0, dtype=np.float32)
        self.load_data(self.data_path)
        self.n_classes = self.classes.shape[0]

        if self.dataset in RGB_DATASETS:
            self.get_statistics()

    def load_data(self, data_path: str) -> None:
        """Load original (full) dataset."""
        if self.dataset == "cifar10":
            dataset = np.load(
                os.path.join(
                    data_path, "training.npz" if self.train else "validation.npz"
                )
            )
            self.images = dataset["data"]
            self.labels = dataset["labels"]

        else:
            dataset = torch.load(
                os.path.join(
                    data_path, "training.pt" if self.train else "validation.pt"
                )
            )
            self.images = dataset[0].numpy()
            self.labels = dataset[1].numpy()
        self.classes = np.unique(self.labels)

    def get_statistics(self) -> None:
        """Get means and stds of CIFAR-10, CIFAR-100, or the ImageNet training data."""
        if self.dataset == "cifar10":
            self.means = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
            self.stds = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
        elif self.dataset == "cifar100":
            self.means = np.array([0.5071, 0.4865, 0.44092], dtype=np.float32)
            self.stds = np.array([0.2673, 0.2564, 0.2761], dtype=np.float32)
        elif self.dataset == "imagenet":
            self.means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            raise Exception(
                "\nWe do not want to apply image transformations to MNIST-like datasets.\n"
            )

    @jaxtyped
    @typechecker
    def normalize(
        self, img: UInt8orFP32[np.ndarray, "h w c"]
    ) -> UInt8orFP32[np.ndarray, "h w c"]:
        img = img / self.max_pixel_value
        img -= self.means
        img /= self.stds
        return img

    def get_instances(self, hist: np_array) -> Dict[int, np_array]:
        class_samples = {}
        for k in self.classes:
            class_partition = np.where(self.labels == k)[0]
            try:
                class_subsample = np.random.choice(
                    class_partition, size=hist[k], replace=False
                )
            except ValueError:
                class_subsample = np.random.choice(
                    class_partition, size=hist[k], replace=True
                )
            class_samples[k] = class_subsample
        return class_samples

    def sample_examples(self, n_classes: int, n_totals: int, p: float) -> np.ndarray:
        class_distribution = self.get_class_distribution(num_classes=n_classes, p=p)
        sample = np.random.choice(
            n_classes, size=n_totals, replace=True, p=class_distribution
        )
        sample = self.add_remainder(sample, n_classes)
        return sample

    @staticmethod
    def get_class_distribution(num_classes: int, p: float, k: int = 3) -> np.ndarray:
        """With probabilities $(p/k)$ and $(1-p)/(T-k)$ sample $k$ frequent and $T-k$ rare classes respectively."""
        distribution = np.zeros(num_classes)
        p_k = p / k
        q_k = (1 - p) / (num_classes - k)
        frequent_classes = np.random.choice(num_classes, size=k, replace=False)
        rare_classes = np.asarray(
            list(set(range(num_classes)).difference(list(frequent_classes)))
        )
        distribution[frequent_classes] += p_k
        distribution[rare_classes] += q_k
        return distribution

    @staticmethod
    def add_remainder(sample: np.ndarray, n_classes: int) -> np.ndarray:
        remainder = np.array(
            [y for y in np.arange(n_classes) if y not in np.unique(sample)]
        )
        sample = np.hstack((sample, remainder))
        return sample

    @staticmethod
    def get_histogram(sample: np.ndarray, min_samples: int) -> np.ndarray:
        _, hist = zip(
            *sorted(Counter(sample).items(), key=lambda kv: kv[0], reverse=False)
        )
        hist = np.array(hist)
        # guarantee that there are at least C (= min_samples) examples per class
        hist = np.where(hist < min_samples, hist + abs(hist - min_samples), hist)
        return hist

    def sample_instances(self) -> Tuple[Dict[int, np_array], np_array]:
        """Randomly sample class instances as determined per our exponential function."""
        n_totals = self.n_samples * self.n_classes
        sample = self.sample_examples(
            n_classes=self.n_classes, n_totals=n_totals, p=self.probability_mass
        )
        hist = self.get_histogram(sample=sample, min_samples=self.min_samples)
        class_instances = self.get_instances(hist)
        return class_instances, hist

    @jaxtyped
    @typechecker
    def get_subset(
        self,
    ) -> Tuple[
        Union[UInt8orFP32[Array, "n h w"], UInt8orFP32[Array, "n h w c"]],
        Float32[Array, "n num_cls"],
    ]:
        """Get a few-shot subset of the data."""
        sampled_instances, _ = self.sample_instances()
        samples = []
        for cls_instances in sampled_instances.values():
            cls_samples = []
            for idx in cls_instances:
                img = self.images[idx]
                label = self.labels[idx]
                if self.dataset in RGB_DATASETS:
                    img = self.normalize(img)
                cls_samples.append((img, label))
            samples.extend(cls_samples)
        images, labels = zip(*samples)
        images = jnp.array(images)
        labels = jnp.array(labels)
        labels = jax.nn.one_hot(x=labels, num_classes=jnp.max(labels) + 1)
        assert images.shape[0] == labels.shape[0]
        return (images, labels)

    @staticmethod
    def reduce_set(N: int, addition: jnp_array) -> jnp_array:
        # return jnp.array(list(filter(lambda i: i not in addition, range(N))))
        reduced_indices = list(range(N))
        for i in addition:
            reduced_indices.pop(reduced_indices.index(i))
        return jnp.array(reduced_indices)

    @staticmethod
    def get_set_addition(
        y_train: Float32[Array, "n num_cls"], val_classes: np_array, seed: int
    ) -> jnp_array:
        return jnp.array(
            [
                jax.random.choice(
                    jax.random.PRNGKey(seed),
                    jnp.where(jnp.nonzero(y_train)[-1] == k)[0],
                ).item()
                for k in range(y_train.shape[-1])
                if k not in val_classes
            ]
        )

    def adjust_splits(self, X_train, y_train, X_val, y_val, val_classes):
        """Adjust train-val splits to make sure that at least one example per class is in the val set."""
        addition = self.get_set_addition(y_train, val_classes, self.seed)
        X_addition = X_train[addition]
        y_addition = y_train[addition]
        # TODO: find a way to add examples of missing classes to the val set without copying from and reducing the train set
        reduced_indices = self.reduce_set(X_train.shape[0], addition)
        X_train_adjusted = X_train[reduced_indices]
        y_train_adjusted = y_train[reduced_indices]
        X_val_adjusted = jnp.concatenate((X_val, X_addition), axis=0)
        y_val_adjusted = jnp.concatenate((y_val, y_addition), axis=0)
        return X_train_adjusted, y_train_adjusted, X_val_adjusted, y_val_adjusted

    @jaxtyped
    @typechecker
    def create_splits(
        self,
        images: Union[UInt8orFP32[Array, "n h w"], UInt8orFP32[Array, "n h w c"]],
        labels: Float32[Array, "n num_cls"],
    ) -> Tuple[
        Tuple[
            Union[
                UInt8orFP32[Array, "n_train h w"], UInt8orFP32[Array, "n_train h w c"]
            ],
            Float32[Array, "n_train num_cls"],
        ],
        Tuple[
            Union[UInt8orFP32[Array, "n_val h w"], UInt8orFP32[Array, "n_val h w c"]],
            Float32[Array, "n_val num_cls"],
        ],
    ]:
        """Split the few-shot subset of the full training data into a train and a validation set."""
        rnd_perm = np.random.permutation(np.arange(images.shape[0]))
        X_train = images[rnd_perm[: int(len(rnd_perm) * self.train_frac)]]
        y_train = labels[rnd_perm[: int(len(rnd_perm) * self.train_frac)]]
        X_val = images[rnd_perm[int(len(rnd_perm) * self.train_frac) :]]
        y_val = labels[rnd_perm[int(len(rnd_perm) * self.train_frac) :]]

        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0]

        train_classes = jnp.unique(jnp.nonzero(y_train)[-1])
        val_classes = jnp.unique(jnp.nonzero(y_val)[-1])

        if len(train_classes) > len(val_classes):
            # make sure that at least one example per class is in the val set
            X_train, y_train, X_val, y_val = self.adjust_splits(
                X_train, y_train, X_val, y_val, val_classes
            )

        return (X_train, y_train), (X_val, y_val)
