#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import math
import os
import pickle
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from ml_collections import config_dict

import models
from data import DataLoader
from training import OOOTrainer

Array = np.ndarray
Array = jnp.ndarray
FrozenDict = config_dict.FrozenConfigDict


def get_combination(
    samples: List[int],
    epochs: List[int],
    batch_sizes: List[int],
    learning_rates: List[float],
    seeds: List[int],
):
    combs = []
    combs.extend(
        list(
            itertools.product(zip(samples, epochs, batch_sizes, learning_rates), seeds)
        )
    )
    # NOTE: for SLURM use "SLURM_ARRAY_TASK_ID"
    return combs[int(os.environ["SGE_TASK_ID"])]


def make_path(
    root: str,
    model_config: FrozenDict,
    data_config: FrozenDict,
    rnd_seed: int,
) -> str:
    path = os.path.join(
        root,
        model_config.task,
        model_config.type + model_config.depth,
        f"{data_config.n_samples}_samples",
        data_config.distribution,
        f"seed{rnd_seed:02d}",
    )
    return path


def create_dirs(
    results_root: str,
    data_config: FrozenDict,
    model_config: FrozenDict,
    rnd_seed: int,
):
    """Create directories for saving and loading model checkpoints."""
    dir_config = config_dict.ConfigDict()
    log_dir = make_path(results_root, model_config, data_config, rnd_seed)
    dir_config.log_dir = log_dir

    if not os.path.exists(log_dir):
        print("\n...Creating results directory.\n")
        os.makedirs(log_dir, exist_ok=True)

    return dir_config


def run(
    model,
    model_config: FrozenDict,
    data_config: FrozenDict,
    optimizer_config: FrozenDict,
    dir_config: FrozenDict,
    train_set: Tuple[Array, Array],
    val_set: Tuple[Array, Array],
    epochs: int,
    steps: int,
    rnd_seed: int,
    inference: bool = False,
) -> tuple:
    trainer = OOOTrainer(
        model=model,
        model_config=model_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
        dir_config=dir_config,
        steps=steps,
        rnd_seed=rnd_seed,
    )
    # TODO: remove inference flag and create separate inference.py file just for inference
    if inference:
        metrics = trainer.merge_metrics()
        epoch = epochs
    else:
        train_batches = DataLoader(
            data=train_set,
            data_config=data_config,
            model_config=model_config,
            seed=rnd_seed,
            train=True,
        )
        val_batches = DataLoader(
            data=val_set,
            data_config=data_config,
            model_config=model_config,
            seed=rnd_seed,
            train=False,
        )
        metrics, epoch = trainer.train(train_batches, val_batches)
    return trainer, metrics, epoch


def batch_inference(
    trainer: object,
    X_test: Array,
    y_test: Array,
    batch_size: int,
) -> Tuple[float, Dict[int, float]]:
    losses = []
    cls_hits = defaultdict(list)
    for i in range(math.ceil(X_test.shape[0] / batch_size)):
        X_i = X_test[i * batch_size : (i + 1) * batch_size]
        y_i = y_test[i * batch_size : (i + 1) * batch_size]
        loss, cls_hits = trainer.eval_step((X_i, y_i), cls_hits=cls_hits)
        losses.append(loss)
    loss = np.mean(losses)
    return loss, cls_hits


def inference(
    out_path: str,
    trainer: object,
    X_test: Array,
    y_test: Array,
    train_labels: Array,
    model_config: FrozenDict,
    data_config: FrozenDict,
    dir_config: FrozenDict,
    distribution: str,
    batch_size: int = None,
    collect_reps: bool = False,
) -> None:
    if distribution == "heterogeneous":
        if collect_reps:
            reps_path = os.path.join(dir_config.log_dir, "reps")
            if not os.path.exists(reps_path):
                os.makedirs(reps_path)
            test_performance, reps, y_hat = trainer.eval_step((X_test, y_test))
            with open(os.path.join(reps_path, "representations.npz"), "wb") as f:
                np.savez_compressed(f, reps=reps, classes=y_test, predictions=y_hat)
        else:
            try:
                loss, cls_hits = trainer.eval_step(
                    (X_test, y_test), cls_hits=defaultdict(list)
                )
            except (RuntimeError, MemoryError):
                warnings.warn(
                    "\nTest set does not fit into the GPU's memory.\nSplitting test set into small batches to counteract memory problems.\n"
                )
                assert isinstance(
                    batch_size, int
                ), "\nBatch size required to circumvent problems with GPU VRAM.\n"
                loss, cls_hits = batch_inference(
                    trainer=trainer,
                    X_test=X_test,
                    y_test=y_test,
                    batch_size=batch_size,
                )
        acc = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
        test_performance = flax.core.FrozenDict({"loss": loss, "accuracy": acc})
        train_labels = jnp.nonzero(train_labels, size=train_labels.shape[0])[-1]
        cls_distribution = dict(Counter(train_labels.tolist()))
    else:
        test_performance = trainer.eval_step((X_test, y_test))

    print(test_performance)
    print()

    save_results(
        out_path=out_path,
        performance=test_performance,
        cls_distribution=cls_distribution,
        model_config=model_config,
        data_config=data_config,
    )


def make_results_df(
    columns: List[str],
    performance: FrozenDict,
    cls_distribution: Dict[int, int],
    model_config: FrozenDict,
    data_config: FrozenDict,
) -> pd.DataFrame:
    results_current_run = pd.DataFrame(index=range(1), columns=columns)
    results_current_run["model"] = model_config.type + model_config.depth
    results_current_run["dataset"] = data_config.name
    results_current_run["class-distribution"] = [cls_distribution]
    results_current_run["class-performance"] = [list(performance["accuracy"].items())]
    results_current_run["avg-performance"] = np.mean(
        list(performance["accuracy"].values())
    )
    results_current_run["cross-entropy"] = performance["loss"]
    results_current_run["training"] = model_config.task
    results_current_run["n_samples"] = data_config.n_samples
    results_current_run["probability"] = data_config.class_probs
    return results_current_run


def save_results(
    out_path: str,
    performance: FrozenDict,
    cls_distribution: Dict[int, int],
    model_config: FrozenDict,
    data_config: FrozenDict,
) -> None:
    if not os.path.exists(out_path):
        print("\nCreating results directory...\n")
        os.makedirs(out_path, exist_ok=True)

    if os.path.isfile(os.path.join(out_path, "results.pkl")):
        print(
            "\nFile for results exists.\nConcatenating current results with existing results file...\n"
        )
        results_overall = pd.read_pickle(os.path.join(out_path, "results.pkl"))
        results_current_run = make_results_df(
            columns=results_overall.columns.values,
            performance=performance,
            cls_distribution=cls_distribution,
            model_config=model_config,
            data_config=data_config,
        )
        results = pd.concat(
            [results_overall, results_current_run], axis=0, ignore_index=True
        )
        results.to_pickle(os.path.join(out_path, "results.pkl"))
    else:
        print("\nCreating file for results...\n")
        columns = [
            "model",
            "dataset",
            "class-distribution",
            "class-performance",
            "avg-performance",
            "cross-entropy",
            "training",
            "n_samples",
            "probability",
        ]
        results_current_run = make_results_df(
            columns=columns,
            performance=performance,
            cls_distribution=cls_distribution,
            model_config=model_config,
            data_config=data_config,
        )
        results_current_run.to_pickle(os.path.join(out_path, "results.pkl"))


def create_model(*, model_cls, model_config):
    platform = jax.local_devices()[0].platform
    if model_config.half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    model = model_cls(
        num_classes=model_config.n_classes,
        task=model_config.task,
        triplet_dim=512 if model_config.task == "mtl" else None,
        dtype=model_dtype,
    )
    return model


def get_model(model_config: FrozenDict, data_config: FrozenDict):
    """Create model instance."""
    model_name = model_config.type + model_config.depth
    net = getattr(models, model_name)
    if model_config.type.lower() == "resnet":
        model = create_model(model_cls=net, model_config=model_config)
    elif model_config.type.lower() == "vit":
        model = net(
            embed_dim=256,
            hidden_dim=512,
            num_heads=8,
            num_layers=3,  # 6
            patch_size=4,
            num_channels=3,
            num_patches=64,
            num_classes=model_config.n_classes,
            dropout_prob=0.2,
            triplet_dim=512 if model_config.task == "mtl" else None,
            task=model_config.task,
            capture_intermediates=False,
        )
    elif model_config.type.lower() == "custom":
        # all layers up to the penultimate layer are conv blocks
        # last layer is fully-connected
        if data_config.name.lower().endswith("mnist"):
            # MNIST, FashionMNIST
            encoder_widths = [32, 64, 128, 128]
        else:
            # CIFAR-10, CIFAR-100, SVHN need a more expressive network
            encoder_widths = [32, 64, 128, 256, 512, 256]
        model = net(
            encoder_widths=encoder_widths,
            num_classes=model_config.n_classes,
            source=data_config.name,
            task=model_config.task,
            triplet_dim=256 if model_config.task == "mtl" else None,
            capture_intermediates=False,
        )
    else:
        raise Exception("\nNo model type other than CNN, ResNet or ViT implemented.\n")
    return model
