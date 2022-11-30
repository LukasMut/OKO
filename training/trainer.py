#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Tuple

import flax
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints
from flax.training.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import training.utils as utils
from training.train_state import TrainState

Array = jnp.ndarray
Model = Any


@dataclass(init=True, repr=True)
class OOOTrainer:
    model: Model
    model_config: FrozenDict
    data_config: FrozenDict
    optimizer_config: FrozenDict
    dir_config: FrozenDict
    steps: int
    rnd_seed: int

    def __post_init__(self) -> None:
        self.rng_seq = hk.PRNGSequence(self.rnd_seed)
        self.rng = jax.random.PRNGKey(self.rnd_seed)
        # freeze model config dictionary (i.e., make it immutable)
        self.model_config = FrozenDict(self.model_config)
        # number of elements in tuple
        self.k = 6
        # inititalize model
        self.init_model()

        self.logger = SummaryWriter(log_dir=self.dir_config.log_dir)
        self.early_stop = EarlyStopping(
            min_delta=1e-4, patience=self.optimizer_config.patience
        )
        # create jitted train and eval functions
        self.create_functions()

        # initialize two empty lists to store train and val performances
        self.train_metrics = list()
        self.test_metrics = list()

    def init_model(self) -> None:
        """Initialise parameters (i.e., weights and biases) of neural network."""
        key_i, key_j = random.split(random.PRNGKey(self.rnd_seed))

        if self.data_config.name.endswith("mnist"):
            H, W = self.data_config.input_dim
            C = 1  # gray channel
        else:
            H, W, C = self.data_config.input_dim

        def get_init_batch(batch_size: int) -> Array:
            return random.normal(key_i, shape=(batch_size * self.k , H, W, C))

        if self.model_config["type"].lower() == "resnet":
            batch = get_init_batch(self.data_config.ooo_batch_size)
            variables = self.model.init(key_j, batch, train=True)
            init_params, self.init_batch_stats = (
                variables["params"],
                variables["batch_stats"],
            )
            setattr(self, "init_params", init_params)
        else:
            if self.model_config["type"].lower() == "vit":
                batch = get_init_batch(self.data_config.ooo_batch_size)
                self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
                init_params = self.model.init(
                    {"params": init_rng, "dropout": dropout_init_rng},
                    batch,
                    train=True,
                )["params"]
                setattr(self, "init__params", init_params)
            else:
                batch = get_init_batch(self.data_config.ooo_batch_size)
                variables = self.model.init(key_j, batch)
                _, init_params = variables.pop("params")
                setattr(self, "init_params", init_params)
                del variables

            self.init_batch_stats = None
        self.state = None

    def get_optim(self, train_batches: Iterator) -> Any:
        opt_class = getattr(optax, self.optimizer_config.name)
        opt_hypers = {}
        # opt_hypers["learning_rate"] = self.optimizer_config.lr
        if self.optimizer_config.name.lower() == "sgd":
            opt_hypers["momentum"] = self.optimizer_config.momentum
            opt_hypers["nesterov"] = True

        # we decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        if self.data_config["name"] == 'cifar10':
            steps = [0.25, 0.5, 0.75]
        else:
            steps = [0.3, 0.6, 0.9]
        schedule = {
            int(len(train_batches) * self.optimizer_config.epochs * steps[0]): 0.1,
            int(len(train_batches) * self.optimizer_config.epochs * steps[1]): 0.1,
            int(len(train_batches) * self.optimizer_config.epochs * steps[2]): 0.1,
        } 
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.optimizer_config.lr,
            boundaries_and_scales=schedule,
        )
        # clip gradients at maximum value
        transf = [optax.clip(1.0)]
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **opt_hypers),
        )
        return optimizer

    def init_optim(self, train_batches: Iterator) -> None:
        """Initialize optimizer and training state."""
        optimizer = self.get_optim(train_batches)
        # initialize training state
        self.state = TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            batch_stats=self.init_batch_stats
            if self.state is None
            else self.state.batch_stats,
            tx=optimizer,
        )

    def create_functions(self) -> None:
        def apply_l2_reg(state: Any, lmbda: float) -> Tuple[Any, Array]:
            weight_penalty, grads = jax.value_and_grad(
                utils.l2_reg, argnums=0, has_aux=False
            )(state.params, lmbda)
            state = state.apply_gradients(grads=grads)
            return state, weight_penalty

        def init_loss_fn(model_config: FrozenDict, state: Any) -> Callable:
            loss_fn = partial(
                getattr(utils, f"loss_fn_{model_config['type'].lower()}"), state
            )
            return loss_fn

        def train_step(model_config: FrozenDict, state, batch: Tuple[Array], rng=None):
            loss_fn = self.init_loss_fn(state)
            # get loss, gradients for objective function, and other outputs of loss function
            if model_config["type"].lower() == "resnet":
                (loss, aux), grads = jax.value_and_grad(
                    loss_fn, argnums=0, has_aux=True
                )(
                    state.params,
                    batch,
                    True,
                )
                # update parameters and batch statistics
                state = state.apply_gradients(
                    grads=grads,
                    batch_stats=aux[1]["batch_stats"],
                )
            else:
                if model_config["type"].lower() == "vit":
                    (loss, aux), grads = jax.value_and_grad(
                        loss_fn, argnums=0, has_aux=True
                    )(
                        state.params,
                        batch,
                        rng,
                        True,
                    )
                    self.rng = aux[1]
                else:
                    (loss, aux), grads = jax.value_and_grad(
                        loss_fn, argnums=0, has_aux=True
                    )(
                        state.params,
                        batch,
                    )
                    state = state.apply_gradients(grads=grads)

            # NOTE: l2-regularization does not appear to be necessary/improve generalizationperformance
            if model_config["regularization"]:
                state, weight_penalty = apply_l2_reg(
                    state=state, lmbda=model_config["weight_decay"]
                )
                loss += weight_penalty

            return state, loss, aux

        def inference(model_config, state, X: Array, rng=None) -> Array:
            if model_config["type"].lower() == "custom":
                logits = utils.cnn_predict(
                    state=state,
                    params=state.params,
                    X=X,
                    train=False,
                )
            elif model_config["type"].lower() == "resnet":
                logits = utils.resnet_predict(
                    state=state,
                    params=state.params,
                    X=X,
                    train=False,
                )
            elif model_config["type"].lower() == "vit":
                logits, _ = utils.vit_predict(
                    state=state,
                    params=state.params,
                    rng=rng,
                    X=X,
                    train=False,
                )
            return logits

        # partially initialize functions
        self.init_loss_fn = partial(init_loss_fn, self.model_config)
        self.train_step = partial(train_step, self.model_config)
        self.inference = partial(inference, self.model_config)

    def eval_step(self, batch: Tuple[Array], cls_hits: Dict[int, int]):
        X, y = batch
        logits = self.inference(self.state, X=X, rng=self.rng)
        loss = optax.softmax_cross_entropy(logits, y).mean()
        batch_hits = utils.class_hits(logits, y)
        acc = self.collect_hits(cls_hits=cls_hits, batch_hits=batch_hits)
        return loss.item(), acc

    def compute_accuracy(
        self, batch: Tuple[Array], aux, cls_hits: Dict[int, int]
    ) -> Array:
        logits = aux[0] if isinstance(aux, tuple) else aux
        _, y = batch
        batch_hits = utils.class_hits(logits, y)
        acc = self.collect_hits(
            cls_hits=cls_hits,
            batch_hits=batch_hits,
        )
        return acc

    @staticmethod
    def collect_hits(
        cls_hits: Dict[int, List[int]], batch_hits: Dict[int, List[int]]
    ) -> Dict[int, List[int]]:
        for cls, hits in batch_hits.items():
            cls_hits[cls].extend(hits)
        return cls_hits

    def train_epoch(self, batches: Iterator, train: bool) -> Tuple[float, float]:
        """Take a step over each mini-batch in the (train or val) generator."""
        cls_hits = defaultdict(list)
        batch_losses = jnp.zeros(len(batches))
        for step, batch in tqdm(enumerate(batches), desc="Batch", leave=False):
            if train:
                self.state, loss, aux = self.train_step(
                    state=self.state, batch=batch, rng=self.rng
                )
                cls_hits = self.compute_accuracy(
                    batch=batch, aux=aux, cls_hits=cls_hits
                )
            else:
                loss, cls_hits = self.eval_step(
                    batch=batch,
                    cls_hits=cls_hits,
                )
            batch_losses = batch_losses.at[step].set(loss)
        cls_accs = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
        avg_batch_acc = np.mean(list(cls_accs.values()))
        avg_batch_loss = jnp.mean(batch_losses)
        return (avg_batch_loss, avg_batch_acc)

    def train(
        self, train_batches: Iterator, val_batches: Iterator
    ) -> Tuple[Dict[str, Tuple[float]], int]:
        self.init_optim(train_batches=train_batches)
        for epoch in tqdm(range(1, self.optimizer_config.epochs + 1), desc="Epoch"):
            train_performance = self.train_epoch(batches=train_batches, train=True)
            self.train_metrics.append(train_performance)

            if epoch % 2 == 0:
                test_performance = self.train_epoch(batches=val_batches, train=False)
                self.test_metrics.append(test_performance)
                # Tensorboard logging
                self.logger.add_scalar(
                    "train/loss", np.asarray(train_performance[0]), global_step=epoch
                )
                self.logger.add_scalar(
                    "train/acc", np.asarray(train_performance[1]), global_step=epoch
                )
                self.logger.add_scalar(
                    "val/loss", np.asarray(test_performance[0]), global_step=epoch
                )
                self.logger.add_scalar(
                    "val/acc", np.asarray(test_performance[1]), global_step=epoch
                )
                print(
                    f"Epoch: {epoch:03d}, Train Loss: {train_performance[0]:.4f}, Train Acc: {train_performance[1]:.4f}, Val Loss: {test_performance[0]:.4f}, Val Acc: {test_performance[1]:.4f}\n"
                )
                self.logger.flush()

            if epoch % self.steps == 0:
                self.save_model(epoch=epoch)

                """
                intermediate_performance = utils.merge_metrics(
                    (self.train_metrics, self.test_metrics))
                utils.save_metrics(out_path=os.path.join(
                    self.out_path, 'metrics'), metrics=intermediate_performance, epoch=f'{epoch+1:04d}')
                """

            if epoch > self.optimizer_config.burnin:
                _, self.early_stop = self.early_stop.update(test_performance[1])
                if self.early_stop.should_stop:
                    print("\nMet early stopping criteria, stopping training...\n")
                    break

        metrics = self.merge_metrics(self.train_metrics, self.test_metrics)
        return metrics, epoch

    @staticmethod
    def merge_metrics(train_metrics, test_metrics) -> FrozenDict:
        return flax.core.FrozenDict(
            {"train_metrics": train_metrics, "test_metrics": test_metrics}
        )

    @staticmethod
    def save_metrics(out_path: str, metrics, epoch: int) -> None:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, f"metrics_{epoch}.pkl"), "wb") as f:
            pickle.dump(metrics, f)

    def save_model(self, epoch: int = 0) -> None:
        # save current model at certain training iteration
        if self.model_config["type"].lower() == "resnet":
            target = {
                "params": self.state.params,
                "batch_stats": self.state.batch_stats,
            }
        else:
            target = self.state.params
        checkpoints.save_checkpoint(
            ckpt_dir=self.dir_config.log_dir, target=target, step=epoch, overwrite=True
        )

    def load_model(self) -> None:
        """Loade model checkpoint. Different checkpoint is used for pretrained models."""
        if self.model_config["type"].lower() == "resnet":
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.dir_config.log_dir, target=None
            )
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                mle_params=state_dict["params"],
                batch_stats=state_dict["batch_stats"],
                tx=self.state.tx
                if self.state
                else optax.sgd(self.optimizer_config.lr, momentum=0.9),
            )
        else:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=self.dir_config.log_dir, target=None
            )
            self.state = TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.state.tx
                if self.state
                else optax.adam(self.optimizer_config.lr),
                batch_stats=None,
            )

    def __len__(self) -> int:
        return self.optimizer_config.epochs
