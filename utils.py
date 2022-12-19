#!/usr/bin/env python3
# -*- coding: utf-8 -*

import os
import pickle
import re
from typing import Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from flax import serialization
from jaxtyping import AbstractDtype, Array, Float32, jaxtyped
from ml_collections import config_dict
from typeguard import typechecked as typechecker

RGB_DATASETS = ["cifar10", "cifar100", "imagenet", "imagenet_lt"]
MODELS = ["Custom", "ResNet18", "ResNet50", "ResNet101", "ViT"]

FrozenDict = config_dict.FrozenConfigDict


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


def get_data(dataset: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    tf_split = get_tf_split(split)
    images, labels = tfds.as_numpy(
        tfds.load(
            dataset,
            split=tf_split,
            batch_size=-1,
            as_supervised=True,
        )
    )
    images = jnp.asarray(images)
    labels = jax.nn.one_hot(x=labels, num_classes=np.unique(labels).shape[0])
    return (images, labels)


def get_tf_split(split: str) -> str:
    if split == "train":
        tf_split = "train[:80%]"
    elif split == "val":
        tf_split = "train[80%:]"
    else:
        tf_split = split
    return tf_split


def get_data_statistics(
    dataset: str,
) -> Tuple[Float32[Array, "3"], Float32[Array, "3"]]:
    """Get means and stds of CIFAR-10, CIFAR-100, or the ImageNet training data."""
    if dataset == "cifar10":
        means = jnp.array([0.4914, 0.4822, 0.4465], dtype=jnp.float32)
        stds = jnp.array([0.2023, 0.1994, 0.2010], dtype=jnp.float32)
    elif dataset == "cifar100":
        means = jnp.array([0.5071, 0.4865, 0.44092], dtype=jnp.float32)
        stds = jnp.array([0.2673, 0.2564, 0.2761], dtype=jnp.float32)
    elif dataset == "imagenet":
        means = jnp.array([0.485, 0.456, 0.406], dtype=jnp.float32)
        stds = jnp.array([0.229, 0.224, 0.225], dtype=jnp.float32)
    else:
        raise Exception(f"\nDataset statistics for {dataset} are not available.\n")
    return means, stds


@jaxtyped
@typechecker
def normalize_images(
    images: UInt8orFP32[Array, "#batchk h w c"],
    data_config: FrozenDict,
) -> UInt8orFP32[Array, "#batchk h w c"]:
    images = images / data_config.max_pixel_value
    images -= data_config.means
    images /= data_config.stds
    return images


def load_metrics(metric_path):
    """Load pretrained parameters into memory."""
    binary = find_binaries(metric_path)
    metrics = pickle.loads(open(os.path.join(metric_path, binary), "rb").read())
    return metrics


def save_params(out_path, params, epoch):
    """Encode parameters of network as bytes and save as binary file."""
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    bytes_output = serialization.to_bytes(params)
    with open(os.path.join(out_path, f"pretrained_params_{epoch}.pkl"), "wb") as f:
        pickle.dump(bytes_output, f)


def save_opt_state(out_path, opt_state, epoch):
    """Encode parameters of network as bytes and save as binary file."""
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    bytes_output = serialization.to_bytes(opt_state)
    with open(os.path.join(out_path, f"opt_state_{epoch}.pkl"), "wb") as f:
        pickle.dump(bytes_output, f)


def find_binaries(param_path):
    """Search for last checkpoint."""
    param_binaries = sorted(
        [
            f
            for _, _, files in os.walk(param_path)
            for f in files
            if re.search(r"(?=.*\d+)(?=.*pkl$)", f)
        ]
    )
    return param_binaries.pop()


def merge_params(pretrained_params, current_params):
    return flax.core.FrozenDict(
        {"encoder": pretrained_params["encoder"], "clf": current_params["clf"]}
    )


def get_subset(y, hist):
    subset = []
    for k, freq in enumerate(hist):
        subset.extend(
            np.random.choice(np.where(y == k)[0], size=freq, replace=False).tolist()
        )
    subset = np.random.permutation(subset)
    return subset
