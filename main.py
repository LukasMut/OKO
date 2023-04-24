#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import math
import os
import random
import warnings
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
from jaxtyping import AbstractDtype, Array, Float32, jaxtyped
from ml_collections import config_dict
from sklearn.metrics import roc_auc_score
from tensorflow_probability.substrates import jax as tfp
from typeguard import typechecked as typechecker

import models
import utils
from config import get_configs
from data import DataPartitioner, OKOLoader
from training import OKOTrainer

FrozenDict = config_dict.FrozenConfigDict


class UInt8orFP32(AbstractDtype):
    dtypes = ["uint8", "float32"]


os.environ["PYTHONIOENCODING"] = "UTF-8"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# NOTE: start out allocating very little memory,
# and as the program gets run and more GPU memory is needed,
# the GPU memory region is extended for the TensorFlow process.
# Memory is not released since it can lead to memory fragmentation.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# NOTE: uncomment line below and comment out lines above if running TensorFlow ops only on CPU
# tf.config.experimental.set_visible_devices([], "GPU")

gpu_devices = jax.local_devices(backend="gpu")


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--out_path", type=str, help="path/to/params")
    aa(
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "fashion_mnist",
            "cifar10",
            "cifar10_corrupted",
            "cifar100",
            "imagenet",
            "imagenet_lt",
            "downsampled_imagenet",
        ],
    )
    aa(
        "--network",
        type=str,
        default="ResNet18",
        choices=utils.MODELS,
        help="choice of feature encoder, f: x \in \mathbb{R}^{h \times w \times c} \to z \in \mathbb{R}^{d}",
    )
    aa("--samples", type=int, nargs="+", help="average number of samples per class")
    aa("--n_classes", type=int, help="number of classes in dataset")
    aa(
        "--k",
        type=int,
        nargs="+",
        choices=list(range(10)),
        help="number of odd classes in a set of k+2 examples with 2 examples coming from the same class",
    )
    aa(
        "--targets",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help="whether to use hard targets with a point mass at the pair class or soft targets that reflect the true class distribution in a set",
    )
    aa(
        "--oko_batch_sizes",
        type=int,
        nargs="+",
        help="number of sets per mini-batch (i.e., number of subsamples x 3",
    )
    aa(
        "--main_batch_sizes",
        type=int,
        nargs="+",
        help="number of triplets per mini-batch (i.e., number of subsamples x 3",
    )
    aa(
        "--num_sets",
        type=int,
        nargs="+",
        help="maximum number of triplets during each epoch",
    )
    aa(
        "--probability_masses",
        type=float,
        nargs="+",
        help="probability mass that will be equally distributed among the k most frequent classes",
    )
    aa("--epochs", type=int, nargs="+", help="maximum number of epochs")
    aa("--etas", type=float, nargs="+", help="learning rate for optimizer")
    aa(
        "--optim",
        type=str,
        default="sgd",
        choices=["adam", "adamw", "radam", "sgd", "rmsprop"],
    )
    aa(
        "--burnin",
        type=int,
        default=30,
        help="burnin period until convergence criterion is evaluated (is equal to the minimum number of epochs)",
    )
    aa(
        "--patience",
        type=int,
        default=20,
        help="Number of steps of no improvement before stopping training",
    )
    aa(
        "--warmup_epochs",
        type=int,
        default=20,
        choices=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        help="number of warmup epochs for linear schedule",
    )
    aa("--steps", type=int, help="save intermediate parameters every <steps> epochs")
    aa(
        "--sampling",
        type=str,
        default="uniform",
        nargs="+",
        choices=["uniform"],
        help="how to sample mini-batches per iteration",
    )
    aa(
        "--min_samples",
        type=int,
        default=None,
        help="minimum number of examples per class",
    )
    aa(
        "--overrepresented_classes",
        type=int,
        default=3,
        help="number of classes that will be overrepresented with probability mass <p>",
    )
    aa(
        "--regularization",
        action="store_true",
        help="apply l2 regularization on the model's params during training",
    )
    aa(
        "--apply_augmentations",
        action="store_true",
        help="use data augmentations during training",
    )
    aa(
        "--label_noise",
        action="store_true",
        help="use data augmentations during training",
    )
    aa(
        "--collect_reps",
        action="store_true",
        help="whether to store encoder latent representations",
    )
    aa(
        "--seeds",
        type=int,
        nargs="+",
        help="list of random seeds for cross-validating results over different random initializations",
    )
    args = parser.parse_args()
    return args


def get_combination(
    samples: List[int],
    epochs: List[int],
    oko_batch_sizes: List[int],
    main_batch_sizes: List[int],
    learning_rates: List[float],
    num_sets: List[int],
    probability_masses: List[float],
    num_odds: List[int],
    sampling_policies: List[str],
    seeds: List[int],
):
    combs = []
    combs.extend(
        list(
            itertools.product(
                zip(
                    samples,
                    epochs,
                    oko_batch_sizes,
                    main_batch_sizes,
                    learning_rates,
                    num_sets,
                ),
                probability_masses,
                num_odds,
                sampling_policies,
                seeds,
            )
        )
    )
    # NOTE: for SLURM use "SLURM_ARRAY_TASK_ID"
    return combs[int(os.environ["SLURM_ARRAY_TASK_ID"])]


def make_log_dir(
    root: str,
    model_config: FrozenDict,
    data_config: FrozenDict,
    rnd_seed: int,
) -> str:
    path = os.path.join(
        root,
        "logs",
        data_config.name,
        model_config.task,
        model_config.type + model_config.depth,
        data_config.targets,
        f"{data_config.n_samples}_samples",
        f"{data_config.class_probs:.2f}",
        f"seed{rnd_seed:02d}",
    )
    return path


def make_calibration_dir(
    root: str,
    model_config: FrozenDict,
    data_config: FrozenDict,
    rnd_seed: int,
) -> str:
    path = os.path.join(
        root,
        "calibration",
        model_config.task,
        model_config.type + model_config.depth,
        data_config.targets,
        f"{data_config.n_samples}_samples",
        f"{data_config.class_probs:.2f}",
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
    log_dir = make_log_dir(results_root, model_config, data_config, rnd_seed)
    dir_config.log_dir = log_dir

    if not os.path.exists(log_dir):
        print("\n...Creating directory to store model checkpoints.\n")
        os.makedirs(log_dir, exist_ok=True)

    return dir_config


@jaxtyped
@typechecker
def get_splits(
    dataset: str,
) -> Tuple[
    Tuple[UInt8orFP32[Array, "n_train h w c"], Float32[Array, "n_train num_cls"]],
    Tuple[UInt8orFP32[Array, "n_val h w c"], Float32[Array, "n_val num_cls"]],
    Tuple[UInt8orFP32[Array, "n_test h w c"], Float32[Array, "n_test num_cls"]],
]:
    train_set = utils.get_data(dataset, split="train")
    val_set = utils.get_data(dataset, split="val")
    test_set = utils.get_data(dataset, split="test")
    return (train_set, val_set, test_set)


@jaxtyped
@typechecker
def get_fs_subset(
    train_set: Tuple[
        UInt8orFP32[Array, "n_train h w c"], Float32[Array, "n_train num_cls"]
    ],
    n_samples: int,
    min_samples: int,
    p_mass: float,
    overrepresented_classes: int,
    rnd_seed: int,
) -> Tuple[UInt8orFP32[Array, "n_prime h w c"], Float32[Array, "n_prime num_cls"]]:
    """Get a subset with <n_samples> data points of the full training data, following a long tail class distribution."""
    data_partitioner = DataPartitioner(
        images=train_set[0],
        labels=train_set[1],
        n_samples=n_samples,
        min_samples=min_samples,
        probability_mass=p_mass,
        overrepresented_classes=overrepresented_classes,
        seed=rnd_seed,
    )
    fs_subset = data_partitioner.partitioning()
    return fs_subset


def run(
    model,
    model_config: FrozenDict,
    data_config: FrozenDict,
    optimizer_config: FrozenDict,
    dir_config: FrozenDict,
    train_set: Tuple[Array, Array],
    val_set: Tuple[Array, Array],
    steps: int,
    rnd_seed: int,
) -> tuple:
    trainer = OKOTrainer(
        model=model,
        model_config=model_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
        dir_config=dir_config,
        steps=steps,
        rnd_seed=rnd_seed,
    )
    train_batches = OKOLoader(
        data=train_set,
        data_config=data_config,
        model_config=model_config,
        seed=rnd_seed,
        train=True,
    )
    val_batches = OKOLoader(
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
    predictions = []
    cls_hits = defaultdict(list)
    for i in range(math.ceil(X_test.shape[0] / batch_size)):
        X_i = X_test[i * batch_size : (i + 1) * batch_size]
        y_i = y_test[i * batch_size : (i + 1) * batch_size]
        loss, cls_hits, logits = trainer.eval_step(X_i, y_i, cls_hits=cls_hits)
        losses.append(loss)
        predictions.append(logits)
    predictions = jnp.vstack(predictions)
    loss = np.mean(losses)
    return loss, cls_hits, predictions


def inference(
    out_path: str,
    epoch: int,
    trainer: OKOTrainer,
    X_test: Float32[Array, "n_test h w c"],
    y_test: Float32[Array, "n_test num_cls"],
    train_labels: Float32[Array, "n_prime num_cls"],
    model_config: FrozenDict,
    data_config: FrozenDict,
    dir_config: FrozenDict,
    batch_size: int = None,
    collect_reps: bool = False,
) -> None:
    X_test = jax.device_put(X_test, device=gpu_devices[0])
    y_test = jax.device_put(y_test, device=gpu_devices[0])
    if collect_reps:
        reps_path = os.path.join(dir_config.log_dir, "reps")
        if not os.path.exists(reps_path):
            os.makedirs(reps_path)
        test_performance, reps, y_hat = trainer.eval_step(X_test, y_test)
        with open(os.path.join(reps_path, "representations.npz"), "wb") as f:
            np.savez_compressed(f, reps=reps, classes=y_test, predictions=y_hat)
    else:
        try:
            loss, cls_hits, logits = trainer.eval_step(
                X_test,
                y_test,
                cls_hits=defaultdict(list),
            )
            probas = jax.nn.softmax(logits)
        except (RuntimeError, MemoryError):
            warnings.warn(
                "\nTest set does not fit into the GPU's memory.\nSplitting test set into small batches and running batch-wise inference to counteract memory problems on current node.\n"
            )
            assert isinstance(
                batch_size, int
            ), "\nBatch size parameter required to circumvent problems with GPU VRAM.\n"
            loss, cls_hits, logits = batch_inference(
                trainer=trainer,
                X_test=X_test,
                y_test=y_test,
                batch_size=batch_size,
            )
            probas = jax.nn.softmax(logits)

    def entropy(p: Float32[Array, "#batch num_cls"]) -> Float32[Array, "#batch"]:
        return -jnp.sum(jnp.where(p == 0, 0, p * jnp.log(p)), axis=-1)

    entropies = entropy(probas)
    avg_entropy = entropies.mean().item()
    acc = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
    auc = roc_auc_score(
        y_true=np.asarray(y_test), y_score=np.asarray(probas), average="macro"
    )
    true_labels = jnp.nonzero(y_test)[-1]
    ece = tfp.stats.expected_calibration_error(
        num_bins=10,
        logits=logits,
        labels_true=true_labels,
    ).item()
    brier_score = tfp.stats.brier_score(labels=true_labels, logits=logits).mean().item()
    uncertainty, resolution, reliability = tfp.stats.brier_decomposition(
        labels=true_labels,
        logits=logits,
    )
    brier_decomp = (uncertainty.item(), resolution.item(), reliability.item())
    test_performance = flax.core.FrozenDict(
        {
            "loss": loss,
            "auc": auc,
            "avg-entropy": avg_entropy,
            "accuracy": acc,
            "brier_score": brier_score,
            "brier_decomp": brier_decomp,
            "ece": ece,
        }
    )
    train_labels = jnp.nonzero(train_labels, size=train_labels.shape[0])[-1]
    cls_distribution = dict(Counter(train_labels.tolist()))

    print(test_performance)
    print()

    save_results(
        out_path=out_path,
        epoch=epoch,
        performance=test_performance,
        cls_distribution=cls_distribution,
        model_config=model_config,
        data_config=data_config,
    )
    return logits, train_labels


def sort_cls_distribution(cls_distribution: Dict[int, int]) -> Dict[int, int]:
    return dict(sorted(cls_distribution.items(), key=lambda kv: kv[1], reverse=True))


def get_cls_subset_performance(
    cls_accuracies: Dict[int, float], cls_subset: List[int]
) -> Tuple[float]:
    _, cls_subset_performances = zip(
        *list(filter(lambda x: x[0] in cls_subset, cls_accuracies))
    )
    return cls_subset_performances


def get_cls_subset_performances(
    cls_distribution: Dict[int, int], cls_accuracies: Dict[int, float], k: int = 3
) -> Tuple[Tuple[float], Tuple[float]]:
    cls_distribution = sort_cls_distribution(cls_distribution)
    classes = list(cls_distribution.keys())
    frequent_classes = classes[:k]
    rare_classes = classes[k:]
    performance_frequent_classes = get_cls_subset_performance(
        cls_accuracies, frequent_classes
    )
    performance_rare_classes = get_cls_subset_performance(cls_accuracies, rare_classes)
    return performance_frequent_classes, performance_rare_classes


def make_results_df(
    columns: List[str],
    epoch: int,
    performance: FrozenDict,
    cls_distribution: Dict[int, int],
    model_config: FrozenDict,
    data_config: FrozenDict,
) -> pd.DataFrame:
    accuracies = list(performance["accuracy"].items())
    (
        performance_frequent_classes,
        performance_rare_classes,
    ) = get_cls_subset_performances(
        cls_distribution=cls_distribution,
        cls_accuracies=accuracies,
    )
    results_current_run = pd.DataFrame(index=range(1), columns=columns)
    results_current_run["model"] = model_config.type + model_config.depth
    results_current_run["dataset"] = data_config.name
    results_current_run["class-distribution"] = [cls_distribution]
    results_current_run["class-performance"] = [accuracies]
    results_current_run["avg-performance-overall"] = np.mean(
        list(map(lambda x: x[1], accuracies))
    )
    results_current_run["avg-performance-frequent-classes"] = np.mean(
        performance_frequent_classes
    )
    results_current_run["avg-performance-rare-classes"] = np.mean(
        performance_rare_classes
    )
    results_current_run["cross-entropy"] = performance["loss"]
    results_current_run["auc"] = performance["auc"]
    results_current_run["ece"] = performance["ece"]
    results_current_run["brier_score"] = performance["brier_score"]
    results_current_run["brier_decomp"] = [performance["brier_decomp"]]
    results_current_run["avg-entropy"] = performance["avg-entropy"]
    results_current_run["training"] = model_config.task
    results_current_run["sampling"] = data_config.sampling
    results_current_run["weighting"] = False
    results_current_run["targets"] = data_config.targets
    results_current_run["n_samples"] = data_config.n_samples * data_config.n_classes
    results_current_run["n_frequent_classes"] = data_config.n_frequent_classes
    results_current_run["min_samples"] = int(data_config.min_samples)
    results_current_run["probability"] = data_config.class_probs
    results_current_run["num_sets"] = data_config.num_sets
    results_current_run["initial_lr"] = data_config.initial_lr
    results_current_run["l2_reg"] = model_config.regularization
    results_current_run["convergence_time"] = epoch
    results_current_run["label_noise"] = data_config.label_noise
    return results_current_run


def save_results(
    out_path: str,
    epoch: int,
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
            epoch=epoch,
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
            "avg-performance-overall",
            "avg-performance-frequent-classes",
            "avg-performance-rare-classes",
            "cross-entropy",
            "auc",
            "ece",
            "brier_score",
            "brier_decomp",
            "avg-entropy",
            "training",
            "sampling",
            "weighting",
            "n_samples",
            "targets",
            "n_frequent_classes",
            "min_samples",
            "num_sets",
            "probability",
            "initial_lr",
            "l2_reg",
            "convergence_time",
            "label_noise",
        ]
        results_current_run = make_results_df(
            columns=columns,
            epoch=epoch,
            performance=performance,
            cls_distribution=cls_distribution,
            model_config=model_config,
            data_config=data_config,
        )
        results_current_run.to_pickle(os.path.join(out_path, "results.pkl"))


def create_model(model_cls, model_config, data_config) -> Any:
    return model_cls(
        num_classes=model_config.n_classes,
        k=data_config.k,
    )


def get_model(model_config: FrozenDict, data_config: FrozenDict):
    """Create model instance."""
    model_name = model_config.type + model_config.depth
    net = getattr(models, model_name)
    if model_config.type.lower() == "resnet":
        model = create_model(
            model_cls=net, model_config=model_config, data_config=data_config
        )
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
            k=data_config.k,
            dropout_prob=0.2,
            capture_intermediates=False,
        )
    elif model_config.type.lower() == "custom":
        # all layers up to the penultimate layer are conv blocks
        # last layer is fully-connected
        if data_config.name.lower().endswith("mnist"):
            # MNIST, FashionMNIST
            encoder_widths = [32, 64, 128, 128]
        else:
            # CIFAR-10 and CIFAR-100 need a more expressive network than MNIST/FashionMNIST
            encoder_widths = [32, 64, 128, 256, 512, 256]
        model = net(
            encoder_widths=encoder_widths,
            num_classes=model_config.n_classes,
            k=data_config.k,
            source=data_config.name,
            capture_intermediates=False,
        )
    else:
        raise ValueError(
            "\nNo model type other than (custom) CNN, ResNet or ViT implemented.\n"
        )
    return model


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # get current combination of settings
    (
        (n_samples, epochs, oko_batch_size, main_batch_size, eta, num_sets),
        p_mass,
        num_odds,
        sampling,
        rnd_seed,
    ) = get_combination(
        samples=args.samples,
        epochs=args.epochs,
        oko_batch_sizes=args.oko_batch_sizes,
        main_batch_sizes=args.main_batch_sizes,
        learning_rates=args.etas,
        num_sets=args.num_sets,
        probability_masses=args.probability_masses,
        num_odds=args.k,
        sampling_policies=args.sampling,
        seeds=args.seeds,
    )

    # seed random number generator
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)

    train_set, val_set, test_set = get_splits(args.dataset)
    train_set = get_fs_subset(
        train_set=train_set,
        min_samples=args.min_samples,
        p_mass=p_mass,
        n_samples=n_samples,
        overrepresented_classes=args.overrepresented_classes,
        rnd_seed=rnd_seed,
    )

    input_dim = train_set[0].shape[1:]
    num_classes = train_set[1].shape[-1]

    data_config, model_config, optimizer_config = get_configs(
        args,
        n_samples=n_samples,
        input_dim=input_dim,
        epochs=epochs,
        oko_batch_size=oko_batch_size,
        main_batch_size=main_batch_size,
        num_sets=num_sets,
        p_mass=p_mass,
        num_odds=num_odds,
        eta=eta,
        sampling=sampling,
    )

    if args.dataset in utils.RGB_DATASETS:
        val_images, val_labels = val_set
        test_images, test_labels = test_set
        val_images = utils.normalize_images(images=val_images, data_config=data_config)
        test_images = utils.normalize_images(
            images=test_images, data_config=data_config
        )
        val_set = (val_images, val_labels)
        test_set = (test_images, test_labels)

    model = get_model(model_config=model_config, data_config=data_config)

    dir_config = create_dirs(
        results_root=args.out_path,
        data_config=data_config,
        model_config=model_config,
        rnd_seed=rnd_seed,
    )

    trainer, metrics, epoch = run(
        model=model,
        model_config=model_config,
        data_config=data_config,
        optimizer_config=optimizer_config,
        dir_config=dir_config,
        train_set=train_set,
        val_set=val_set,
        steps=args.steps,
        rnd_seed=rnd_seed,
    )
    logits, train_labels = inference(
        out_path=args.out_path,
        epoch=epoch,
        trainer=trainer,
        X_test=test_set[0],
        y_test=test_set[1],
        train_labels=train_set[1],
        model_config=model_config,
        data_config=data_config,
        dir_config=dir_config,
        batch_size=main_batch_size,
        collect_reps=args.collect_reps,
    )
    calibration_dir = make_calibration_dir(
        root=args.out_path,
        data_config=data_config,
        model_config=model_config,
        rnd_seed=rnd_seed,
    )
    if not os.path.exists(calibration_dir):
        print("\n...Creating directory for analyzing model calibration.\n")
        os.makedirs(calibration_dir, exist_ok=True)

    with open(os.path.join(calibration_dir, "labels_plus_probas.npz"), "wb") as f:
        np.savez_compressed(
            file=f,
            train_labels=np.array(train_labels),
            test_labels=np.array(test_set[1]),
            test_logits=np.array(logits),
        )
