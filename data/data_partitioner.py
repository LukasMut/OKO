#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = ["DataPartitioner"]

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torchvision.transforms as T

import utils

np_array = np.ndarray
jnp_array = jnp.ndarray


RGB_DATASETS = ["cifar10", "cifar100", "imagenet"]


@dataclass(init=True, repr=True)
class DataPartitioner:
    dataset: str
    data_path: str
    n_samples: int
    distribution: str
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

        self.load_data(self.data_path)
        self.n_classes = self.classes.shape[0]

        if self.dataset in RGB_DATASETS:
            self.transform = self.get_transform()

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
                    data_path, "training.pt" if self.train else "validation.npz"
                )
            )
            self.images = dataset[0].numpy()
            self.labels = dataset[1].numpy()
        self.classes = np.unique(self.labels)

    @staticmethod
    def get_statistics(dataset: str) -> Tuple[List[float], List[float]]:
        """Get means and STDs of training data for CIFAR-10/CIFAR-100."""
        if dataset == "cifar10":
            means = [0.4914, 0.4822, 0.4465]
            stds = [0.24703, 0.24349, 0.26159]
            # stds = [0.2023, 0.1994, 0.2010]
        elif dataset == "cifar100":
            means = [0.5071, 0.4865, 0.44092]
            stds = [0.2673, 0.2564, 0.2761]
        elif dataset == "imagenet":
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
        else:
            raise Exception(
                "\nWe do not want to apply image transformations to MNIST-like datasets.\n"
            )
        return means, stds

    def get_transform(self) -> Any:
        """Compose image data transformations."""
        means, stds = self.get_statistics(self.dataset)
        transform = T.Compose(
            [T.ToPILImage(), T.ToTensor(), T.Normalize(mean=means, std=stds)]
        )
        return transform

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

    def sample_instances(self) -> Tuple[Dict[int, np_array], np_array]:
        """Randomly sample class instances as determined per our exponential function."""
        n_totals = self.n_samples * self.n_classes
        sample = utils.sample_instances(
            n_classes=self.n_classes, n_totals=n_totals, p=self.probability_mass
        )
        hist = utils.get_histogram(sample, self.min_samples)
        class_instances = self.get_instances(hist)
        return class_instances, hist

    def get_subset(self) -> Tuple[jnp_array, jnp_array]:
        """Get a few-shot subset of the data."""
        sampled_instances, _ = self.sample_instances()
        samples = []
        for cls_instances in sampled_instances.values():
            cls_samples = []
            for idx in cls_instances:
                img = self.images[idx]
                label = self.labels[idx]
                if hasattr(self, "transform"):
                    img = self.transform(img).permute(1, 2, 0).numpy()
                cls_samples.append((img, label))
            samples.extend(cls_samples)
        images, labels = zip(*samples)
        images = jnp.array(images)
        labels = jnp.array(labels)
        # create one-hot labels
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
        y_train: jnp_array, val_classes: np_array, seed: int
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

    def create_splits(
        self, images, labels
    ) -> Tuple[Tuple[jnp_array, jnp_array], Tuple[jnp_array, jnp_array]]:
        """Construct train and validation splits of the few-shot data subset."""
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

    def save_subsets(
        self, train_set: jnp_array, val_set: jnp_array, out_path: str
    ) -> None:
        """Save few-shot subset to disk."""
        out_path = self.make_outpath(out_path)
        filename = f"few_shot_{self.dataset}.hdf5"
        with h5py.File(os.path.join(out_path, filename), "w") as fp:
            fp["train_images"] = train_set[0]
            fp["train_labels"] = train_set[1]
            fp["val_images"] = val_set[0]
            fp["val_labels"] = val_set[1]

    def make_outpath(self, out_path: str) -> str:
        """Create outout directory."""
        out_path = os.path.join(
            out_path,
            f"{self.n_samples:d}_samples",
            self.distribution,
            f"seed{self.seed:02d}",
        )
        if not os.path.exists(out_path):
            print(f"\n...Creating output directory: {out_path}\n")
            os.makedirs(out_path, exist_ok=True)
        try:
            os.remove(os.path.join(out_path, f"ooo_dataset_{self.dataset}.hdf5"))
        except FileNotFoundError:
            print("\nThere is no file to be removed.\n")
        return out_path
