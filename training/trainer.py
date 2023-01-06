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
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float32, PyTree
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

import training.loss_funs as loss_funs
from training.train_state import TrainState


@dataclass(init=True, repr=True)
class OptiMaker:
    dataset: str
    epochs: int
    optimizer: str
    lr: float
    clip_val: float
    momentum: float = None

    def get_optim(self, num_batches: int) -> Any:
        opt_class = getattr(optax, self.optimizer)
        opt_hypers = {}
        # opt_hypers["learning_rate"] = self.optimizer_config.lr
        if self.optimizer.lower() == "sgd":
            opt_hypers["momentum"] = self.momentum
            opt_hypers["nesterov"] = True

        # we decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        steps = [0.25, 0.5, 0.75] if self.dataset == "cifar10" else [0.3, 0.6, 0.9]
        schedule = {
            int(num_batches * self.epochs * steps[0]): 0.1,
            int(num_batches * self.epochs * steps[1]): 0.1,
            int(num_batches * self.epochs * steps[2]): 0.1,
        }
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=self.lr,
            boundaries_and_scales=schedule,
        )
        # clip gradients at maximum value
        transf = [optax.clip(self.clip_val)]
        optimizer = optax.chain(
            *transf,
            opt_class(lr_schedule, **opt_hypers),
        )
        return optimizer


@dataclass(init=True, repr=True)
class OKOTrainer:
    model: PyTree
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
        # inititalize model
        self.init_model()

        self.logger = SummaryWriter(log_dir=self.dir_config.log_dir)
        self.early_stop = EarlyStopping(
            min_delta=1e-4, patience=self.optimizer_config.patience
        )
        # create jitted train and eval functions
        self.create_functions()

        self.optimaker = OptiMaker(
            self.data_config.name,
            self.optimizer_config.epochs,
            self.optimizer_config.name,
            self.optimizer_config.lr,
            self.optimizer_config.clip_val,
            self.optimizer_config.momentum,
        )

        self.gpu_devices = jax.local_devices(backend="cpu")

        # initialize two empty lists to store train and val performances
        self.train_metrics = list()
        self.test_metrics = list()

    def init_model(self) -> None:
        """Initialise parameters (i.e., weights and biases) of neural network."""
        key_i, key_j = random.split(random.PRNGKey(self.rnd_seed))
        H, W, C = self.data_config.input_dim

        def get_init_batch(batch_size: int) -> Float32[Array, "#batchk h w c"]:
            return random.normal(
                key_i, shape=(batch_size * (self.data_config.k + 2), H, W, C)
            )

        if self.model_config["type"].lower() == "resnet":
            batch = get_init_batch(self.data_config.oko_batch_size)
            variables = self.model.init(key_j, batch, train=True)
            init_params, self.init_batch_stats = (
                variables["params"],
                variables["batch_stats"],
            )
            setattr(self, "init_params", init_params)
        else:
            if self.model_config["type"].lower() == "vit":
                batch = get_init_batch(self.data_config.oko_batch_size)
                self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
                init_params = self.model.init(
                    {"params": init_rng, "dropout": dropout_init_rng},
                    batch,
                    train=True,
                )["params"]
                setattr(self, "init_params", init_params)
            else:
                batch = get_init_batch(self.data_config.oko_batch_size)
                variables = self.model.init(key_j, batch)
                _, init_params = variables.pop("params")
                setattr(self, "init_params", init_params)
                del variables

            self.init_batch_stats = None
        self.state = None

    def init_optim(self, num_batches: int) -> None:
        """Initialize optimizer and training state."""
        optimizer = self.optimaker.get_optim(num_batches)
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
        def init_loss_fn(model_config: FrozenDict, state: PyTree) -> Callable:
            loss_fn = partial(
                getattr(loss_funs, f"loss_fn_{model_config['type'].lower()}"), state
            )
            return loss_fn

        @partial(jax.jit, static_argnames=["lmbda"], donate_argnums=[1])
        def apply_l2_reg(
            lmbda: float, state: PyTree
        ) -> Tuple[PyTree, Float32[Array, ""]]:
            weight_penalty, grads = jax.value_and_grad(
                loss_funs.l2_reg, argnums=0, has_aux=False
            )(state.params, lmbda)
            state = state.apply_gradients(grads=grads)
            return state, weight_penalty

        @partial(jax.jit, static_argnames=["loss_fun"], donate_argnums=[1])
        def jit_grads_resnet(
            loss_fun: Callable,
            state: PyTree,
            X: Float32[Array, "#batchk h w c"],
            y: Float32[Array, "#batch num_cls"],
        ):
            (loss, (logits, stats)), grads = jax.value_and_grad(
                loss_fun, argnums=0, has_aux=True
            )(state.params, X, y, True)
            # update parameters and batch statistics
            state = state.apply_gradients(
                grads=grads,
                batch_stats=stats["batch_stats"],
            )
            return state, loss, logits

        @partial(jax.jit, static_argnames=["loss_fun", "rng"], donate_argnums=[1])
        def jit_grads_vit(
            loss_fun: Callable,
            state: PyTree,
            X: Float32[Array, "#batchk h w c"],
            y: Float32[Array, "#batch num_cls"],
            rng,
        ):
            (loss, (logits, rng)), grads = jax.value_and_grad(
                loss_fun, argnums=0, has_aux=True
            )(state.params, X, y, rng, True)
            state = state.apply_gradients(grads=grads)
            return state, loss, logits, rng

        @partial(jax.jit, static_argnames=["loss_fn"], donate_argnums=[1])
        def jit_grads(
            loss_fn,
            state: PyTree,
            X: Float32[Array, "#batchk h w c"],
            y: Float32[Array, "#batch num_cls"],
        ):
            (loss, logits), grads = jax.value_and_grad(
                loss_fn, argnums=0, has_aux=True
            )(state.params, X, y)
            state = state.apply_gradients(grads=grads)
            return state, loss, logits

        def train_step(
            model_config: FrozenDict,
            state: PyTree,
            X: Float32[Array, "#batchk h w c"],
            y: Float32[Array, "#batch num_cls"],
            rng=None,
        ):
            loss_fn = self.init_loss_fn(state)
            # get loss, gradients for objective function, and other outputs of loss function
            if model_config["type"].lower() == "resnet":
                state, loss, logits = partial(jit_grads_resnet, loss_fn)(state, X, y)
            else:
                if model_config["type"].lower() == "vit":
                    state, loss, logits, rng = partial(jit_grads_vit, loss_fn)(
                        state, X, y, rng
                    )
                    self.rng = rng
                else:
                    state, loss, logits = partial(jit_grads, loss_fn)(state, X, y)
            # apply a small amount of l2 regularization
            if model_config["regularization"]:
                state, weight_penalty = partial(
                    apply_l2_reg, model_config["weight_decay"]
                )(state)
                loss += weight_penalty
            return state, loss, logits

        def inference(
            model_config, state: PyTree, X: Float32[Array, "#batchk h w c"], rng=None
        ) -> Array:
            if model_config["type"].lower() == "custom":
                logits = loss_funs.cnn_predict(state, state.params, X, False)
            elif model_config["type"].lower() == "resnet":
                logits = loss_funs.resnet_predict(state, state.params, X, False)
            elif model_config["type"].lower() == "vit":
                logits, _ = loss_funs.vit_predict(state, state.params, X, rng, False)
            return logits

        # partially initialize functions
        self.init_loss_fn = partial(init_loss_fn, self.model_config)
        self.train_step = partial(train_step, self.model_config)
        self.inference = partial(inference, self.model_config)

    def eval_step(
        self,
        X: Float32[Array, "#batchk h w c"],
        y: Float32[Array, "#batch num_cls"],
        cls_hits: Dict[int, int],
    ):
        logits = self.inference(self.state, X=X, rng=self.rng)
        loss = optax.softmax_cross_entropy(logits, y).mean()
        batch_hits = loss_funs.class_hits(logits, y)
        acc = self.collect_hits(cls_hits=cls_hits, batch_hits=batch_hits)
        return loss.item(), acc, logits

    def compute_accuracy(
        self,
        y: Float32[Array, "#batch num_cls"],
        logits: Float32[Array, "#batch num_cls"],
        cls_hits: Dict[int, int],
    ) -> Array:
        batch_hits = loss_funs.class_hits(logits, y)
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
            X, y = tuple(jax.device_put(x, device=self.gpu_devices[0]) for x in batch)
            if train:
                self.state, loss, logits = self.train_step(
                    state=self.state, X=X, y=y, rng=self.rng
                )
                cls_hits = self.compute_accuracy(y=y, logits=logits, cls_hits=cls_hits)
            else:
                loss, cls_hits, _ = self.eval_step(
                    X=X,
                    y=y,
                    cls_hits=cls_hits,
                )
            del X
            del y
            batch_losses = batch_losses.at[step].set(loss)
        cls_accs = {cls: np.mean(hits) for cls, hits in cls_hits.items()}
        avg_batch_acc = np.mean(list(cls_accs.values()))
        avg_batch_loss = jnp.mean(batch_losses)
        return (avg_batch_loss, avg_batch_acc)

    def train(
        self, train_batches: Iterator, val_batches: Iterator
    ) -> Tuple[Dict[str, Tuple[float]], int]:
        self.init_optim(num_batches=len(train_batches))
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
                intermediate_performance = loss_funs.merge_metrics(
                    (self.train_metrics, self.test_metrics))
                loss_funs.save_metrics(out_path=os.path.join(
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
