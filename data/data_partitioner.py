#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataPartitioner"]

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import AbstractDtype, Array, Float32, jaxtyped
from typeguard import typechecked as typechecker


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


class FP32orFP64(AbstractDtype):
    dtypes = ["float32", "float64"]


@dataclass(init=True, repr=True)
class DataPartitioner:
    images: Union[
        UInt8orFP32[np.ndarray, "n_train h w c"], UInt8orFP32[np.ndarray, "n_train h w"]
    ]
    labels: FP32orFP64[np.ndarray, "n_train num_cls"]
    n_samples: int
    probability_mass: float
    overrepresented_classes: int
    min_samples: int
    seed: int

    def __post_init__(self) -> None:
        # seed rng
        random.seed(self.seed)
        np.random.seed(self.seed)
        assert isinstance(
            self.min_samples, int
        ), "\nMinimum number of samples per class must be defined.\n"
        # n_train x num_cls -> n_train x 1
        self.labels = jnp.nonzero(self.labels)[-1]
        self.classes = np.unique(self.labels)
        self.n_classes = self.classes.shape[0]

    def get_instances(
        self, hist: FP32orFP64[np.ndarray, "num_cls"]
    ) -> Dict[int, FP32orFP64[np.ndarray, "_"]]:
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

    def sample_examples(
        self, n_classes: int, n_totals: int, p: float
    ) -> FP32orFP64[np.ndarray, "n_totals"]:
        class_distribution = self.get_class_distribution(
            num_classes=n_classes,
            p=p,
            overrepresented_classe=self.overrepresented_classes,
        )
        sample = np.random.choice(
            n_classes, size=n_totals, replace=True, p=class_distribution
        )
        sample = self.add_remainder(sample, n_classes)
        return sample

    @staticmethod
    def get_class_distribution(
        num_classes: int, p: float, overrepresented_classes: int = 3
    ) -> FP32orFP64[np.ndarray, "num_cls"]:
        """With probabilities $(p/k)$ and $(1-p)/(T-k)$ sample $k$ frequent and $T-k$ rare classes respectively."""
        distribution = np.zeros(num_classes)
        p_k = p / overrepresented_classes
        q_k = (1 - p) / (num_classes - overrepresented_classes)
        frequent_classes = np.random.choice(
            num_classes, size=overrepresented_classes, replace=False
        )
        underrepresented_classes = np.asarray(
            list(set(range(num_classes)).difference(list(frequent_classes)))
        )
        distribution[frequent_classes] += p_k
        distribution[underrepresented_classes] += q_k
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

    def sample_instances(
        self,
    ) -> Tuple[
        Dict[int, FP32orFP64[np.ndarray, "_"]], FP32orFP64[np.ndarray, "num_cls"]
    ]:
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
    def partitioning(
        self,
    ) -> Tuple[UInt8orFP32[Array, "n_prime h w c"], Float32[Array, "n_prime num_cls"]]:
        """Get a subset with <n_samples> of the full training data, following a long tail class distribution."""
        sampled_instances, _ = self.sample_instances()
        samples = []
        for cls_instances in sampled_instances.values():
            cls_samples = []
            for idx in cls_instances:
                img = self.images[idx]
                label = self.labels[idx]
                cls_samples.append((img, label))
            samples.extend(cls_samples)
        images, labels = zip(*samples)
        images = jnp.array(images)
        labels = jnp.array(labels)
        labels = jax.nn.one_hot(x=labels, num_classes=jnp.unique(labels).shape[0])
        assert images.shape[0] == labels.shape[0]
        return (images, labels)
